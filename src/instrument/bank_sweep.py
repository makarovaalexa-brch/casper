r"""bank_sweep.py -- concept-vs-item at matched budget under FOUR static question banks, one data build.

Each bank is a different answer to "which question is worth asking?", applied identically to both
channels so the comparison is a channel comparison and not a selection-policy comparison:

  pop      most answerable            (item: play count; concept: member mass)
  entropy  4-way H incl. unanswerable  -- DEGENERATE on a sparse catalogue, collapses onto pop
  polar    H(bands | answered)         -- pure discrimination, ignores whether anyone can answer
  expinfo  P(answered) * H(bands|ans)  -- the defensible middle

The item arm and the concept arm share users, hidden targets, budgets and metric; an unanswered
question BURNS the budget in both. At m=0 both arms fold nothing and must score identically -- that
equality is the arms-matched canary and is asserted.

  python src/instrument/bank_sweep.py --ckpt .cache/instrument/cv3_ep11.pt
"""
import os
import sys
import json
import argparse

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "src", "baselines"))

import concept_distill_train as CD
from concept_fold import ConceptFoldNet
from concept_cold_check import (build_banks, cold_interviews, cold_item_interviews, score, MS, log)

BANKS = ("pop", "entropy", "polar", "expinfo")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(_ROOT, ".cache", "instrument", "cv3_ep11.pt"))
    ap.add_argument("--snapshot", default=os.path.join(_ROOT, "models", "i26c_ep31", "t2i26c_ep31.pt"))
    ap.add_argument("--n_users", type=int, default=1500)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--max_users", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(_ROOT, "experiments", "instrument",
                                                  "bank_sweep.json"),
                    help="output JSON. Pass a NEW filename when the protocol changes -- the "
                         "2026-08-05 policy-B pool moves every cell, and the pre-B artifact must "
                         "stay readable next to the doc that quotes it.")
    a = ap.parse_args()
    a.lam = 0.0; a.arm = "s1"; a.tau = 1.0; a.eps_l2 = 0.0; a.m_max = 16

    S = CD._build_train_stack(a)
    mc = np.array([len(S["member_sets"][i]) for i in range(len(S["tags"]))], np.float64)
    from train_tower_t2 import compute_head_mask
    import metrics as M
    tr = M.load_train(S["ni"], os.path.join(_ROOT, "data", "ml-25m", "proc"))
    head_mask, _ = compute_head_mask(tr, S["ni"])
    pop = np.asarray(tr.sum(axis=0)).ravel()

    banks = build_banks(S)
    banks["item_pop"] = np.argsort(-pop)
    banks["conc_pop"] = np.argsort(-mc)

    blob = torch.load(a.ckpt, map_location="cpu")
    # Read the ARCHITECTURE AND THE ANSWER CONTRACT from the checkpoint, never from CLI defaults:
    # cv5 is hidden=1024 with graded meh, and scoring it with 256 / zeroed meh either crashes (as it
    # did) or, worse, silently measures a different system.
    net = ConceptFoldNet(len(S["tags"]), d=S["enc"].d_out, h=blob.get("hidden", a.hidden),
                         conc_init=S["d_c"].numpy(), use_refusal=blob.get("use_refusal", False))
    net.load_state_dict(blob["net"])
    keep_meh = bool(blob.get("keep_meh", False))
    log(f"concept net = {os.path.basename(a.ckpt)} ep{blob['epoch']} "
        f"hidden={blob.get('hidden', a.hidden)} keep_meh={keep_meh}")

    out = {}
    for b in BANKS:
        ci = score(None, S, cold_item_interviews(S, banks[f"item_{b}"], n_users=a.n_users), head_mask)
        cc = score(net, S, cold_interviews(S, mc, n_users=a.n_users, rank=banks[f"conc_{b}"], keep_meh=keep_meh), head_mask)
        assert abs(ci[0][0] - cc[0][0]) < 1e-9, "ARMS-MATCHED CANARY FAILED: m=0 must be identical"
        out[b] = {"item": ci, "conc": cc}
        log(f"--- bank={b} (m=0 canary {ci[0][0]:.4f}/{ci[0][1]:.4f} OK)")
        for m in MS[1:]:
            d = cc[m][1] - ci[m][1]
            log(f"    m{m:2d}  conc {cc[m][0]:.4f}/{cc[m][1]:.4f}   item {ci[m][0]:.4f}/{ci[m][1]:.4f}"
                f"   tail_delta {d:+.4f} {'CONC' if d > 0 else 'item'}")
    payload = {b: {k: {str(m): v[m] for m in MS} for k, v in r.items()} for b, r in out.items()}
    payload["_pool_policy"] = ("B -- every rated fold-in-side item of the user (likes AND dislikes, "
                               "held-out targets excluded) is masked, identically for the item arm "
                               "and the concept arm at every budget (2026-08-05 ruling)")
    json.dump(payload, open(a.out, "w"), indent=2)
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
