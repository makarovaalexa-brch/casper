r"""dre_train_ours.py -- STEP P/X part 1: train DRE's seed sets ON OUR RULER.

DRE = Kweon, Kang, Hwang & Yu, "Deep Rating Elicitation for New Users in Collaborative Filtering",
WWW 2020, DOI 10.1145/3366423.3380042. Native snap PASSED 2026-08-06 (ML-1M k=50, NDCG@10
.5593 +/- .0088 vs published .5688) -- see docs/results/SNAP3_DRE_NATIVE.md. This file is the PORT:
same method, our data.

WHAT DRE IS, IN OUR VOCABULARY. The encoder is a (n_items x k) matrix of logits `logpi`; a
Gumbel-Softmax over the ITEM axis makes each of the k columns a near-one-hot selection vector, and
`x @ softmax((logpi+g)/T)` is therefore "the user's ratings on k softly-selected items". A decoder
reconstructs the full catalogue from those k numbers. Annealing T from 10 to 0.1 hardens the
selection; at the end `argmax_i logpi[:, j]` is the item chosen for slot j. So DRE learns a FIXED
length-k seed set end to end: in our terms it is a STATIC selector, and it is the peer-reviewed
learned-static bar.

THE k-DEPENDENCE. The seed set is a function of k -- the columns are trained jointly, so the k=2 set
is NOT the first two items of the k=16 set. One trained artifact per budget, hence this script.

============================================================================================
WHAT WAS ADAPTED FROM `dre_native_snap.py`, AND WHAT WAS NOT
============================================================================================
NOT CHANGED (imported, not re-typed -- `Encoder`, `Decoder`, `dense`, `eval_masked` come straight
from dre_native_snap so the two paths cannot drift):
  * the model: uniform `logpi` init, Gumbel(0,1) noise, softmax over the ITEM axis, `x @ y`;
    decoder Linear(k,300)-Sigmoid-Linear(300,m)-Sigmoid.
  * the losses: MSE for the joint phase; summed elementwise MSE + lambda * sum of L2 parameter norms
    for the decoder-retrain phase.
  * the optimisers and learning rates: Adam 2e-3 joint, Adam 1e-3 retrain, lambda 1e-4.
  * the temperature schedule: geometric T0=10 -> TE=0.1 across the joint phase.
  * batch sizes: 128 joint, 256 retrain.
  * the two-phase structure: train encoder+decoder jointly, take the argmax seed set, then throw the
    decoder away and RETRAIN it from scratch on the hard seed columns (paper Sec 4.3).
  * the binarisation: DRE binarises at rating > 3.5. So does the canonical Liang split. `train.csv`
    IS that matrix; nothing is re-thresholded here.
  * NO duplicate guard. DRE takes a per-column argmax with no constraint that the k columns pick
    distinct items, and we do not add one -- that defect is a property of the published method and
    is measured, not repaired (see `n_duplicate_seeds` in every artifact).

CHANGED, and why:
  1. DATA. `data/ml-25m/proc/train.csv` via `metrics.load_train` -- the canonical Liang split,
     140,768 TRAIN users x 18,359 items. The native script's internal 72/8/20 slice is DROPPED
     because our split already has disjoint train/val/test USER cohorts; using it would carve a
     sub-split out of a cohort that is already the training cohort. Validation and test users are
     never opened by this file (asserted: it only ever reads train.csv).
  2. EPOCH COUNT -> STEP MATCHING. The published recipe is 2000 joint epochs / 150 retrain epochs on
     4,340 ML-1M train users. Replayed literally on 140,768 users that is 32x the gradient steps and
     ~12 CPU-days per budget. We instead match the number of OPTIMISER STEPS:
         joint  : 2000 ep x ceil(4340/128) = 68,000 steps  -> ceil(68000*128/140768) =  62 epochs
         retrain:  150 ep x ceil(4340/256) =  2,550 steps  -> ceil( 2550*256/140768) =   5 epochs
     Step matching keeps the optimiser trajectory length and the annealing ENDPOINT (T=0.1, i.e. a
     hard selection) identical to the snapped recipe; only the number of passes over a much larger
     corpus differs. `--epochs`/`--retrain_epochs` override; both the requested and the realised
     counts are written into the artifact so no cell can be read without them.
  3. STORAGE. The native script densifies the whole matrix; a dense 140,768 x 18,359 float32 tensor
     is 10.3 GB. We hold scipy CSR and densify per batch. Identical arithmetic (the snap already did
     this for ML-20M).
  4. NO INTERNAL EVALUATION. The native script reports DRE's own NDCG on its own held-out slice.
     Here the evaluation is our ruler, run later by `adaptive_harness`, so this script produces
     ARTIFACTS ONLY: the seed set, the retrained decoder, and the duplicate count.

PERSISTENCE (CLAUDE.md HARD RULE 11): each budget is a separate artifact written the moment it
finishes; a re-run skips budgets already on disk and refuses to reuse an artifact whose training
contract differs. The run log is stamped with a unique label so two launches cannot share a file.

USAGE
  python scripts/baselines/dre_train_ours.py --budgets 1,2,4,8,16
  python scripts/baselines/dre_train_ours.py --budgets 2 --epochs 1 --retrain_epochs 1 --tag _SMOKE
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "src", "baselines"))

from dre_native_snap import Encoder, Decoder, dense        # noqa: E402  -- THE method, unmodified
import metrics as M                                        # noqa: E402

PROC = os.path.join(_ROOT, "data", "ml-25m", "proc")
CKDIR = os.path.join(_ROOT, ".cache", "baselines")
LOGDIR = os.path.join(_ROOT, "experiments", "baselines")
os.makedirs(CKDIR, exist_ok=True)
os.makedirs(LOGDIR, exist_ok=True)

# The published recipe, on the corpus it was published for. Only used to derive the step budget.
NATIVE = {"train_users": 4340, "epochs": 2000, "batch": 128,
          "retrain_epochs": 150, "retrain_batch": 256}
N_TRAIN_USERS = 140768
N_ITEMS = 18359


def ckpt_path(k, tag=""):
    return os.path.join(CKDIR, "dre_ml25m_liang_k%d%s.pt" % (k, tag))


# ------------------------------------------------------------------------------------------ data
def load_train_only(log):
    """TRAIN USERS ONLY. `metrics.load_train` reads train.csv and nothing else; validation_*.csv and
    test_*.csv are never touched by this module (grep it -- there is no other loader call)."""
    n_items = len(open(os.path.join(PROC, "unique_sid.txt")).read().split())
    X = M.load_train(n_items, PROC).tocsr().astype(np.float32)
    X.data[:] = 1.0                       # binary >3.5 already; make the contract explicit
    assert X.shape == (N_TRAIN_USERS, N_ITEMS), (
        "split drift: expected (%d, %d), got %s" % (N_TRAIN_USERS, N_ITEMS, X.shape))
    log("[dre] train matrix %s nnz=%d (%.1f items/user), binarised >3.5 by the Liang split"
        % (X.shape, X.nnz, X.nnz / X.shape[0]))
    return X


def step_matched(n_users, batch, native_users, native_epochs, native_batch):
    steps = native_epochs * math.ceil(native_users / native_batch)
    return max(1, math.ceil(steps * batch / n_users)), steps


# ------------------------------------------------------------------------------- train one budget
def train_budget(X, k, args, log):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    n, m = X.shape
    E = Encoder(m, k)
    De = Decoder([args.hidden, k], m)
    crit = nn.MSELoss()
    oE = optim.Adam(E.parameters(), lr=args.lr)
    oD = optim.Adam(De.parameters(), lr=args.lr)
    nb = math.ceil(n / args.batch)
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        # Geometric anneal across the REALISED epoch count: the endpoint T=0.1 (a hard selection) is
        # what the method needs, and it is preserved exactly.
        T = args.T0 * ((args.TE / args.T0) ** (epoch / args.epochs))
        E.train(); De.train()
        tot = 0.0
        for lo in range(0, n, args.batch):                 # shuffle=False, as in the notebook
            x = dense(X, lo, min(lo + args.batch, n))
            oE.zero_grad(); oD.zero_grad()
            loss = crit(De(E(x, T)), x)
            loss.backward(); oE.step(); oD.step()
            tot += float(loss)
            b_i = lo // args.batch
            if args.progress_every and b_i and b_i % args.progress_every == 0:
                el = time.time() - t0
                done = (epoch - 1) * nb + b_i
                log("      [k=%d] ep %d/%d batch %d/%d  %.3fs/step  eta_total=%.1fh"
                    % (k, epoch, args.epochs, b_i, nb, el / done,
                       (el / done) * (args.epochs * nb - done) / 3600.0))
        log("    [k=%d] joint ep %d/%d T=%.5f train_mse=%.6f (%.0fs, %.2fs/step)"
            % (k, epoch, args.epochs, T, tot / nb, time.time() - t0,
               (time.time() - t0) / (epoch * nb)))

    logpi = E.logpi.detach().clone()
    itemseed = torch.max(logpi, dim=0)[1].tolist()          # DRE's own rule; NO duplicate guard
    n_dup = len(itemseed) - len(set(itemseed))
    # Global preference ranking, used ONLY to pad a wasted turn in the harness (see dre_selector).
    pref = logpi.max(dim=1).values
    pad_order = torch.argsort(pref).tolist()[:64]           # DRE's LEAST-preferred items
    log("    [k=%d] seed set = %s  distinct=%d  DUPLICATE SEEDS = %d"
        % (k, itemseed, len(set(itemseed)), n_dup))
    # CONVERGENCE GATE (added 2026-08-08). Distinct argmaxes do NOT prove the selection distribution
    # committed -- a diffuse logpi can still yield k different argmaxes, in which case the seeds are
    # near-arbitrary and the cell measures noise. The 2026-08-07 run could not be checked after the
    # fact because only the resulting itemseed was persisted, so record the distribution itself.
    with torch.no_grad():
        P = torch.softmax(logpi.double(), dim=0)              # softmax over the ITEM axis, per slot
        mx = P.max(dim=0).values                              # (k,) peak probability per slot
        ent = -(P * P.clamp_min(1e-300).log()).sum(dim=0)     # (k,) entropy in nats
        unif = float(math.log(logpi.shape[0]))
        verdict = ("HARD" if float(mx.min()) > 0.9 else
                   "DIFFUSE -- SEEDS ARE NEAR-ARBITRARY, DO NOT REPORT THIS CELL"
                   if float(mx.mean()) < 0.5 else "MIXED")
        sel_stats = {"max_prob_min": float(mx.min()), "max_prob_mean": float(mx.mean()),
                     "entropy_mean_nats": float(ent.mean()), "entropy_max_nats": float(ent.max()),
                     "uniform_entropy_nats": unif, "verdict": verdict}
    log("    [k=%d] SELECTION GATE: max-prob min=%.4f mean=%.4f | entropy mean=%.4f max=%.4f "
        "(uniform=%.3f) -> %s" % (k, sel_stats["max_prob_min"], sel_stats["max_prob_mean"],
                                  sel_stats["entropy_mean_nats"], sel_stats["entropy_max_nats"],
                                  unif, verdict))
    if n_dup:
        log("    [k=%d] *** %d of %d selected columns are DUPLICATES -- the published method has no "
            "duplicate guard and those questions carry zero extra information ***" % (k, n_dup, k))

    # ---- phase 2: throw the decoder away and retrain it on the HARD seed columns (paper Sec 4.3)
    De = Decoder([args.hidden, k], m)
    crit2 = nn.MSELoss(reduction="none")
    oD = optim.Adam(De.parameters(), lr=args.retrain_lr)
    idx = np.arange(n)
    t1 = time.time()
    for epoch in range(args.retrain_epochs):
        rng = np.random.RandomState(args.seed * 1000 + epoch)
        rng.shuffle(idx)
        De.train()
        run = 0.0
        for lo in range(0, n, args.retrain_batch):
            b = np.sort(idx[lo:lo + args.retrain_batch])
            y = torch.from_numpy(np.asarray(X[b].todense(), dtype=np.float32))
            x = y[:, itemseed]
            oD.zero_grad()
            l2 = sum(W.norm(2) for W in De.parameters())
            l = torch.sum(crit2(De(x), y)) + l2 * args.lamb
            l.backward(); oD.step()
            run += float(l)
        log("    [k=%d] retrain ep %d/%d loss=%.1f (%.0fs)"
            % (k, epoch + 1, args.retrain_epochs, run / math.ceil(n / args.retrain_batch),
               time.time() - t1))
    De.eval()

    return {"k": k, "itemseed": [int(i) for i in itemseed],
            "n_duplicate_seeds": int(n_dup), "n_distinct_seeds": int(len(set(itemseed))),
            "pad_order": [int(i) for i in pad_order],
            "selection_gate": sel_stats,
            "decoder": {kk: v.detach().clone() for kk, v in De.state_dict().items()},
            "hidden": int(args.hidden), "n_items": int(m), "n_train_users": int(n),
            "joint_seconds": round(t1 - t0, 1), "retrain_seconds": round(time.time() - t1, 1)}


# ------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", default="1,2,4,8,16")
    ap.add_argument("--epochs", type=int, default=0, help="0 = step-matched to the published recipe")
    ap.add_argument("--retrain_epochs", type=int, default=0, help="0 = step-matched")
    ap.add_argument("--hidden", type=int, default=300)
    ap.add_argument("--T0", type=float, default=10.0)
    ap.add_argument("--TE", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--retrain_lr", type=float, default=0.001)
    ap.add_argument("--retrain_batch", type=int, default=256)
    ap.add_argument("--lamb", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=98765, help="the project's canonical seed")
    ap.add_argument("--tag", default="", help="artifact/log suffix; use _SMOKE for non-citable runs")
    ap.add_argument("--threads", type=int, default=0, help="0 = cpu_count-1")
    ap.add_argument("--progress_every", type=int, default=100,
                    help="log a within-epoch progress line every N batches (0 = off). A 62-epoch "
                         "budget is ~9 CPU-hours, so an epoch-granular log is not a progress trace.")
    args = ap.parse_args()
    torch.set_num_threads(args.threads or max(1, (os.cpu_count() or 4) - 1))

    smoke = "SMOKE" in args.tag.upper()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    lpath = os.path.join(LOGDIR, "DRE_TRAIN_OURS%s_%s.log" % (args.tag, stamp))  # unique: no twin can share it
    lf = open(lpath, "w", encoding="utf-8")

    def log(s):
        print(s, flush=True); lf.write(s + "\n"); lf.flush()

    budgets = [int(x) for x in args.budgets.split(",")]
    if args.epochs == 0:
        args.epochs, _s = step_matched(N_TRAIN_USERS, args.batch, NATIVE["train_users"],
                                       NATIVE["epochs"], NATIVE["batch"])
    if args.retrain_epochs == 0:
        args.retrain_epochs, _s = step_matched(N_TRAIN_USERS, args.retrain_batch,
                                               NATIVE["train_users"], NATIVE["retrain_epochs"],
                                               NATIVE["retrain_batch"])

    log("=" * 100)
    if smoke:
        log("*** SMOKE RUN -- REDUCED EPOCH COUNT, CODE-PATH CHECK ONLY, NOT CITABLE ***")
    log("DRE port training | budgets=%s | joint %d ep x %d steps | retrain %d ep | seed %d"
        % (budgets, args.epochs, math.ceil(N_TRAIN_USERS / args.batch), args.retrain_epochs,
           args.seed))
    log("log -> %s" % lpath)

    contract = hashlib.sha1(json.dumps(
        {kk: v for kk, v in sorted(vars(args).items())
         if kk not in ("budgets", "tag", "threads", "progress_every")}
    ).encode()).hexdigest()[:12]
    log("training contract %s" % contract)

    X = load_train_only(log)

    for k in budgets:
        cp = ckpt_path(k, args.tag)
        if os.path.exists(cp):
            old = torch.load(cp, map_location="cpu")
            if old.get("contract") == contract:
                log("[dre] k=%d already trained under this contract -> skip (%s)" % (k, cp))
                continue
            log("[dre] k=%d exists under contract %s != %s -- retraining"
                % (k, old.get("contract"), contract))
        log("--- k=%d ---" % k)
        blob = train_budget(X, k, args, log)
        blob.update(contract=contract, config=vars(args), seed=args.seed,
                    smoke=smoke, NOT_CITABLE=smoke, stamp=stamp,
                    source="Kweon WWW'20 DRE, ported to the canonical Liang ML-25M train split",
                    trained_on="data/ml-25m/proc/train.csv (train users only)")
        torch.save(blob, cp)                                  # HARD RULE 11: write on completion
        log("[dre] k=%2d WROTE %s | dup_seeds=%d distinct=%d | joint %.0fs + retrain %.0fs"
            % (k, os.path.basename(cp), blob["n_duplicate_seeds"], blob["n_distinct_seeds"],
               blob["joint_seconds"], blob["retrain_seconds"]))

    log("[dre] DUPLICATE-SEED LEDGER (report in every cell):")
    for k in budgets:
        cp = ckpt_path(k, args.tag)
        if os.path.exists(cp):
            b = torch.load(cp, map_location="cpu")
            log("    k=%2d  duplicates=%d  distinct=%d  seeds=%s"
                % (k, b["n_duplicate_seeds"], b["n_distinct_seeds"], b["itemseed"]))
    log("[dre] done")
    lf.close()


if __name__ == "__main__":
    main()
