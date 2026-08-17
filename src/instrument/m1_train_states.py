r"""M1 Stage A4 -- TRAIN-SIDE taste states and concept endorsement.

Author, 2026-08-17: *"1200 limit can be applied to train, or less, but on train you could get actual
taste clusters"*.

WHY TRAIN
---------
Clustering is a FITTING step, and the settled recipe (design sheet D.6b) is fit on the 140,768 TRAIN
users, select/measure on validation. Clustering the 10,000 validation users capped us at ~8 clusters
by the cohort floor alone, which forced blobs of 1,900-3,200 users that no single attribute can
describe. On train, k=30 averages ~4,700 members -- above any floor and small enough to be coherent.

NDCG IS NEVER COMPUTED HERE. The tower was trained on these users, so any NDCG on them is
memorisation (D.6b). What this file takes from train users is only functions of their OWN ratings:
the fold of their liked items, and which concepts they behaviourally endorse under the SAME SEL rule
the validation env applies (`foldable & V>0`, from `signed_values`). No held-out target is read.

PERSISTENCE (HARD RULE 11): each 20k-user slice is written the moment it completes and is resumed
from on restart; a changed contract refuses to resume.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)

import argparse
import json
import time

import numpy as np
import torch
from scipy import sparse

OUT_DIR = os.path.join(_ROOT, "experiments", "battery", "m1", "train")
SEED = 98765
N_TRAIN = 140768
LIKED_MIN = 7
LIKE_LEVEL = 9


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="M1 A4 -- train taste states + endorsement")
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--i_have_author_goahead", action="store_true")
    a = ap.parse_args()
    assert a.i_have_author_goahead, "REFUSING TO RUN without --i_have_author_goahead"
    torch.set_num_threads(a.threads); np.random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    from strategy_channel_suite import build_greedy_env, CERT_TOWER, CERT_CONC
    from concept_fold import fold_items_frozen
    from i26_encoder import UNSEEN_LEVEL

    contract = {"v": 1, "tower": CERT_TOWER, "conc": CERT_CONC, "n_train": N_TRAIN,
                "chunk": a.chunk, "liked_min": LIKED_MIN, "like_level": LIKE_LEVEL,
                "rule": "endorse = foldable & (V>0), the env's own SEL rule"}
    cpath = os.path.join(OUT_DIR, "contract.json")
    if os.path.exists(cpath):
        old = json.load(open(cpath))
        assert old == contract, f"CONTRACT CHANGED -- refusing to resume:\n{old}\n{contract}"
    else:
        json.dump(contract, open(cpath, "w"), indent=1)

    log("#" * 100)
    log("# M1 STAGE A4 -- TRAIN taste states + endorsement. NO NDCG IS COMPUTED ON TRAIN USERS.")
    log("#" * 100)

    for lo in range(0, N_TRAIN, a.chunk):
        hi = min(lo + a.chunk, N_TRAIN)
        out = os.path.join(OUT_DIR, f"slice_{lo}_{hi}.npz")
        if os.path.exists(out):
            log(f"[slice {lo}:{hi}] already banked -- skipping")
            continue
        t0 = time.time()
        E = build_greedy_env(CERT_TOWER, "train", CERT_CONC, f"m1a4_{lo}", cohort_slice=(lo, hi))
        ctx, lvl = E["ctx"], E["lvl"]
        nloc = ctx.n
        seqs, nliked = [], np.zeros(nloc, np.int32)
        for r in range(nloc):
            d = lvl[r]
            ids = [int(i) for i, L in d.items() if LIKED_MIN <= int(L) < UNSEEN_LEVEL]
            nliked[r] = len(ids)
            seqs.append((np.asarray(ids, np.int64),
                         np.full(len(ids), LIKE_LEVEL, np.int64)))
        Z = fold_items_frozen(ctx.enc, seqs, batch=500).numpy().astype(np.float32)
        endorse = sparse.csr_matrix((np.asarray(E["foldable"]) & (np.asarray(E["V"]) > 0)))
        np.savez_compressed(out, z=Z, n_liked=nliked,
                            end_indptr=endorse.indptr, end_indices=endorse.indices,
                            lo=lo, hi=hi, C=E["C"])
        log(f"[slice {lo}:{hi}] {nloc} users · liked mean {nliked.mean():.1f} · "
            f"endorsed/user {endorse.getnnz(1).mean():.1f} · {(time.time()-t0)/60:.1f} min")
        del E, ctx, Z, endorse

    log("[done] all slices banked under " + OUT_DIR)


if __name__ == "__main__":
    main()
