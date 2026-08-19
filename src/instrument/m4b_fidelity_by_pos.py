r"""M4b DIAGNOSTIC -- teacher-forced agreement PER POSITION, on the held-out TRAIN slice.

WHY THIS EXISTS. Stage 2 reported mean dev top-1 0.9997 across seven teachers. Stage 3 then found,
on VALIDATION, that the policy agrees with its teacher at the root for 100% of users and then drops
to 0.63-0.90 at the first branching decision. Those two facts are hard to hold together: at position
1 the policy matched at position 0 for every user, so the rollout state at position 1 IS the
teacher's state and no compounding has happened yet. Rollout agreement at position 1 must therefore
equal TEACHER-FORCED agreement at position 1.

So one of two things is true, and they call for different write-ups:

  (a) the drop is present on TRAIN states too, and the 0.9997 headline hid it by averaging over
      fifteen positions of which most are a fixed tail the teacher had no choice about; or
  (b) the drop is absent on TRAIN states, and train-user states are genuinely easier than
      validation-user states -- which would mean the distillation set is not exchangeable with the
      deployment set, plausibly because the tower was TRAINED on the train users, so their latents
      are in-sample for the encoder and validation users' are not.

This script settles it: same checkpoint, same teacher-forced protocol as stage 2, but reported per
position instead of pooled. No training, no NDCG, nothing written to the paper's ruler.
"""
from __future__ import annotations

import os
import sys

_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _H)

import argparse
import json

import numpy as np
import torch

from m4b_policy import (OUT, SEED, TOTAL_L, TREES, N_TRAIN, consolidate, build_index, make_batch,
                        log)


def main():
    ap = argparse.ArgumentParser(description="per-position teacher-forced agreement on train dev")
    ap.add_argument("--ckpt", default=os.path.join(OUT, "m4b_policy.pt"))
    ap.add_argument("--dev_lo", type=int, default=120000)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--i_have_author_goahead", action="store_true")
    a = ap.parse_args()
    assert a.i_have_author_goahead, "REFUSING TO RUN without --i_have_author_goahead"
    torch.set_num_threads(a.threads)

    from strategy_channel_suite import build_greedy_env, CERT_TOWER, CERT_CONC
    from ndcg_tree import build_pool
    from policy_net import Policy, build_phi

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    banks = {t: consolidate(t) for t in TREES}
    E = build_greedy_env(CERT_TOWER, "train", CERT_CONC, "m4bfid", cohort_slice=(0, 20000))
    pool, _i, _c = build_pool(E, 500, 500)
    assert [list(q) for q in pool] == ck["pool"], "pool moved since training"
    ext = [tuple(q) for q in ck["ext"]]
    norm = (np.asarray(ck["norm"][0], np.float32), np.asarray(ck["norm"][1], np.float32))
    phi_ext, _ = build_phi(E, ext, norm=norm)
    del E
    n_pool = ck["n_pool"]
    d_z = ck["d_z"]
    pol = Policy(d_z, ck["d_phi"], n_pool, arch=ck["arch"], state=ck["state"], use_pos=True)
    pol.load_state_dict(ck["sd"])
    pol.eval()
    phi_cand = torch.from_numpy(phi_ext[:n_pool])

    idx = build_index(banks, a.dev_lo, N_TRAIN)
    log(f"[dev] {len(idx):,} held-out TRAIN examples (users >= {a.dev_lo:,})")
    out = {}
    for ti, t in enumerate(TREES):
        sel = idx[idx[:, 0] == ti]
        by = []
        for p in range(TOTAL_L):
            sub = sel[sel[:, 2] == p]
            if len(sub) == 0:
                by.append(float("nan"))
                continue
            hit = tot = 0
            with torch.no_grad():
                for s in range(0, len(sub), 2048):
                    bb = sub[s:s + 2048]
                    z, hp, ho, hm, ps, am, y = make_batch(banks, phi_ext, bb, ck["a_phi"],
                                                          ck["a_pool"], n_pool, d_z)
                    pr = pol(z, hp, ho, hm, ps, phi_cand, am).argmax(1)
                    hit += int((pr == y).sum())
                    tot += len(bb)
            by.append(hit / tot)
        out[t] = by
        log(f"  [{t}] " + " ".join(f"{v:.3f}" for v in by) +
            f"   pooled {np.nanmean(by):.4f}")
    json.dump({"analysis": "m4b_fidelity_by_position", "split": "train_dev",
               "test_touched": False, "ckpt_epoch": ck["epoch"], "by_position": out},
              open(os.path.join(OUT, "m4b_fidelity_by_pos.json"), "w"), indent=1)
    log("[done]")


if __name__ == "__main__":
    main()
