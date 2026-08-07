r"""dpe_neg_cap_control.py -- the admission control for `adaptive_harness --concept_neg_cap 0`.

WHY THIS EXISTS. Task #18 (DPE) must be differenced per user against the greedy concepts-only
sequence, and the only in-harness re-score of that sequence under the CERTIFIED-ERA answer convention
is `convbandit_greedy_rescore.py`'s NO_NEG_CAP run
(`experiments/baselines/interview/convbandit_rescore_nocap_per_user.npz`). Pairing against it is legal
ONLY if `adaptive_harness --concept_neg_cap 0` produces literally the same numbers as that script's
path. "The means look close" is not that proof -- that error has now been caught twice in this
project.

WHAT IT DOES. Drives the SAME fixed greedy concepts-only action order through `adaptive_harness`'s
own code path (its `_concept_stack` builder at `--concept_neg_cap 0`, its `cap_negatives`-over-tokens
consumer, its `score_arm` call) over the first N test users, and asserts the resulting per-user
full/tail NDCG vectors equal the first N entries of the banked rescore vectors EXACTLY.

Truncating the cohort is legitimate HERE and only here: this is a code-path identity check, not a
measurement, and per-user NDCG is a function of that user's own fold-in, pool and targets alone --
which is exactly what the assert then verifies. Nothing it prints is a citable number.

  python src/baselines/dpe_neg_cap_control.py --users 200
"""
import os
import sys
import json
import argparse

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "src", "instrument"))

import interview_strategies as ST                                          # noqa: E402
import adaptive_harness as AH                                              # noqa: E402
import conversational_bandit_selectors as CB                               # noqa: E402
from arm_n import load_arm_n                                              # noqa: E402
from run_interview_table import (build_recommenders, score_arm, mostpop_row, OUT,  # noqa: E402
                                 answers_to_csr, logln)

SNAP = os.path.join(_ROOT, ".cache", "instrument", "t2i26c_ep31.pt")
CONC_CKPT = os.path.join(_ROOT, "models", "i26c_ep31", "concept_cv3_ep08.pt")
GREEDY = os.path.join(_ROOT, "experiments", "battery", "greedy_static_i26c.json")
REF = os.path.join(OUT, "convbandit_rescore_nocap_per_user.npz")
BUDGETS = (1, 2, 4, 8, 16)


class _Args:
    """The subset of the harness CLI that `_concept_stack` reads."""
    snapshot = SNAP
    keyterm_repr = "uniform_member_mean"
    keyterm_norm_match = 1
    keyterm_n_rep = 20
    concept_neg_cap = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=200)
    a = ap.parse_args()
    assert os.path.exists(REF), f"reference vectors missing: {REF}"

    gj = json.load(open(GREEDY))
    assert gj["tower"] == "t2i26c_ep31.pt" and gj["concept_ckpt"] == "concept_cv3_ep08.pt"
    order_c = [int(c) for c, _i in
               [(i, c) for c, i in gj["arms"]["concepts-only"]["sequence"]]] or None
    seq = [(int(c), int(i)) for c, i in gj["arms"]["concepts-only"]["sequence"]]
    assert len(seq) == 16 and all(c == 1 for c, _ in seq)

    D = load_arm_n(log=logln)
    ni = D["n_items"]
    glob, bank = ST.build_orders(D["g_train"], D["train"], ni, log=logln)
    D = AH._smoke_slice(D, a.users, log=logln)
    n = D["te_tr"].shape[0]
    lvl = ST.level_lookup(D["g_te_tr"])
    built = build_recommenders(["ours"], D, arch="i26", snapshot=SNAP, log=logln)
    pop_row, _prior = mostpop_row(D, log=logln)

    cache = {}
    stack = AH._concept_stack(D, _Args, cache, Y=None, log=logln)
    assert stack["apply_neg_cap"] is False, "the harness built the WRONG answer convention"
    conc = {"n_concepts": int(stack["n_concepts"]),
            "answer": (lambda u, c: CB.concept_answer(stack, u, c))}
    order = [ni + int(i) for _c, i in seq]

    from concept_fold import ConceptFoldNet
    from signed_answers import cap_negatives
    import torch
    blob = torch.load(CONC_CKPT, map_location="cpu")
    sd = blob["net"]
    _nc, _d = tuple(sd["emb.weight"].shape)
    net = ConceptFoldNet(_nc, d=_d, h=int(sd["gate.0.weight"].shape[0]),
                         use_refusal=blob.get("use_refusal", False))
    net.load_state_dict(sd)
    net.eval()

    mode, obj = built["ours"]
    e_, W_, b_ = obj
    ref = np.load(REF, allow_pickle=False)
    worst = 0.0
    for k in BUDGETS:
        sel = AH.Selector()
        sel.name = "greedy-concepts-only"
        sel.select = lambda t, us, _a, _b, _o=order: np.full(len(us), _o[t], dtype=np.int64)
        asked, answered, info = AH.run_interview(sel, k, n, ni, lvl,
                                                 ctx={"D": D, "glob": glob, "bank": bank},
                                                 log=lambda *_: None, conc=conc)
        cs = info["_conc_state"]
        toks = [[(c, v2, r) for (c, _v, r), v2 in
                 zip(t, cap_negatives([x[1] for x in t]))] for t in cs["tokens"]]
        Lc = answers_to_csr(answered, n, ni, mode, asked=asked)
        s_obj = CB.make_concept_aware_predict(e_, W_, b_, Lc, toks, net)
        cell = score_arm("ours_conc", mode, s_obj, asked, answered, D, pop_row, no_floor=True,
                         global_asked=asked[0], per_user=True)
        for ax in ("full", "tail"):
            got = np.asarray(cell[f"{ax}_per_user"], dtype=np.float64)
            # The tail vector covers only the users who HAVE a tail target, so its length is not n.
            # evaluate() walks users in cohort order, so the first len(got) banked entries are the
            # same users -- which the exact-equality assert below then confirms.
            want = ref[f"rescore|greedy-concepts-only|k{k}|{ax}"][:len(got)]
            d = float(np.abs(got - want).max())
            worst = max(worst, d)
            logln(f"[control] k={k:2d} {ax}: mean {got.mean():.4f} vs banked {want.mean():.4f}; "
                  f"max per-user |diff| = {d:.3e}")
    logln(f"[control] {'PASS' if worst == 0.0 else 'FAIL'}: worst per-user diff {worst:.3e} "
          f"over {n} users -- adaptive_harness --concept_neg_cap 0 "
          f"{'IS' if worst == 0.0 else 'IS NOT'} the certified-era convention "
          f"convbandit_greedy_rescore.py measured")
    assert worst == 0.0, "DPE may NOT be differenced against the banked greedy concept vectors"


if __name__ == "__main__":
    main()
