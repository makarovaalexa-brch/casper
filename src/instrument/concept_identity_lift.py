r"""concept_identity_lift.py -- does the trained concept fold use WHICH concept it was given?

THE QUESTION. V2 claims the concept channel is a real channel: folding concept c moves c's own
members. The existing evidence is (i) the concept DIRECTION ranks c's members (whitened AUC 0.94,
measured before any fold) and (ii) folding beats a same-norm shift along the popularity axis
(+0.0087). Neither pins down that the OPERATOR uses phi_c's identity. A gated fold that read the
answer value y as a generic "this user is being positive" nudge would pass both while ignoring the
concept entirely.

PRE-REGISTERED DESIGN (2026-07-29).
  Cohort      : the g5_split sample -- first n_sample rows of the build cohort that have a
                diversified top concept (sh["sel_div"]), pop-matched non-members via sh["get_match"].
  Arms, per (user r, that user's top concept c), each folded from the SAME empty item state:
    LIKE     fold (c, +|v|)              -- the user's own signed value, forced positive
    DISLIKE  fold (c, -|v|)              -- same magnitude, opposite sign
    OTHER    fold (c', +|v|)             -- a random other concept, same value, scored on c's members
  Metric      : member-vs-matched-non-member AUC on c's members, per (user, concept).
  Contrasts   : SIGN     = AUC(LIKE) - AUC(DISLIKE)   -- does the sign of the answer steer members?
                IDENTITY = AUC(LIKE) - AUC(OTHER)     -- does WHICH concept was folded matter?
  Statistics  : paired per-pair bootstrap 95% CI (the same routine as every other CI in the chapter).
  MDE / call  : we care about |delta| > 0.02 AUC. The channel is concept-specific and sign-aware iff
                BOTH contrasts are positive with a CI excluding zero. Report whatever it says --
                a null here is a real finding about the operator, not a bug to hunt.

WHY THIS AND NOT RAW POST-FOLD MEMBER AUC. Raw post-fold member lift is NOT the right bar: the
additive-union operator scores 0.820 member AUC and craters to 0.0991 NDCG (below the no-answer
intercept), while the gated fold scores 0.630 and reaches 0.1759. Pushing all of a concept's members
up the ranking is genre-filter behaviour, which costs accuracy. The differential contrasts above ask
the question that matters -- is the movement concept-specific -- without rewarding that failure mode.

  python src/instrument/concept_identity_lift.py [--n_sample 2000] [--seed 0]
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
from tradeoff_ledger import build_shared, Rung, bootstrap_ci
from concept_fold import ConceptFoldNet
import torch

OUT = os.path.join(_ROOT, "experiments", "battery", "concept_identity_lift.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=PA.SNAP_DEFAULT)
    ap.add_argument("--sclite_ckpt", default=os.path.join(_ROOT, ".cache", "instrument",
                                                          "cd_s1_l10_best.pt"))
    ap.add_argument("--n_sample", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.RandomState(args.seed)

    ctx = build_real_ctx(args.snapshot)
    sh = build_shared(ctx)
    blob = torch.load(args.sclite_ckpt, map_location="cpu")
    net = ConceptFoldNet(len(blob["tags"]), d=ctx.d, h=blob["hidden"])
    net.load_state_dict(blob["net"]); net.eval()
    rung = Rung("sclite", ctx, sh, clite_net=net, val_source="signed")
    log(f"[cid] fold = {os.path.basename(args.sclite_ckpt)}")

    sel = sh["sel_div"]
    rows = [r for r in range(ctx.n) if ctx.va_te[r].nnz > 0 and r in sel][:args.n_sample]
    Wd, bd = ctx.Wd, ctx.bd
    empty1 = (np.empty(0, np.int64), np.empty(0, np.int64))
    n_conc = len(sh["tags"])
    log(f"[cid] {len(rows)} users, {n_conc} concepts, pop-matched non-members")

    def auc_on(S, ti):
        """member-vs-matched-non-member AUC for concept ti, verbatim from g5_split."""
        mem = sh["members"][sh["tags"][ti]]; non = sh["get_match"](ti)
        sm = S[mem]; sn = S[non]; k = len(sm)
        ranks = np.argsort(np.argsort(np.concatenate([sm, sn])))[:k].sum()
        return (ranks - k * (k - 1) / 2) / (k * k)

    # per user: top concept c, its |value|, and a random other concept c'
    tops, vals, others = [], [], []
    for r in rows:
        c = int(sel[r][0][0]); v = abs(float(sel[r][1][0]))
        o = int(rng.randint(n_conc))
        while o == c:
            o = int(rng.randint(n_conc))
        tops.append(c); vals.append(v); others.append(o)

    def fold_scores(concept_ids, values):
        lists = [rung.map_answers(r, [(int(c), float(v))]) or [(int(c), 0.0)]
                 for r, c, v in zip(rows, concept_ids, values)]
        Z = rung.z_batch([empty1] * len(rows), lists)
        return Z

    with torch.no_grad():
        Z_like = fold_scores(tops, [+v for v in vals])
        Z_dis = fold_scores(tops, [-v for v in vals])
        Z_oth = fold_scores(others, [+v for v in vals])

        a_like = np.empty(len(rows)); a_dis = np.empty(len(rows)); a_oth = np.empty(len(rows))
        for j, c in enumerate(tops):
            a_like[j] = auc_on((Z_like[j] @ Wd.T + bd).numpy(), c)
            a_dis[j] = auc_on((Z_dis[j] @ Wd.T + bd).numpy(), c)
            a_oth[j] = auc_on((Z_oth[j] @ Wd.T + bd).numpy(), c)

    sign_d, sign_ci = bootstrap_ci(a_like - a_dis)
    iden_d, iden_ci = bootstrap_ci(a_like - a_oth)
    out = {
        "ckpt": os.path.basename(args.sclite_ckpt),
        "n_users": len(rows), "seed": args.seed,
        "member_AUC": {"like": float(a_like.mean()), "dislike": float(a_dis.mean()),
                       "other_concept": float(a_oth.mean())},
        "SIGN_contrast": {"delta": sign_d, "ci95": sign_ci,
                          "excludes_zero": bool(sign_ci[0] > 0 or sign_ci[1] < 0)},
        "IDENTITY_contrast": {"delta": iden_d, "ci95": iden_ci,
                              "excludes_zero": bool(iden_ci[0] > 0 or iden_ci[1] < 0)},
        "mde": 0.02,
        "seconds": round(time.time() - t0, 1),
    }
    out["PASS"] = bool(sign_d > 0 and sign_ci[0] > 0 and iden_d > 0 and iden_ci[0] > 0)
    json.dump(out, open(OUT, "w"), indent=2)
    log(f"[cid] member AUC  like={a_like.mean():.4f}  dislike={a_dis.mean():.4f}  "
        f"other-concept={a_oth.mean():.4f}")
    log(f"[cid] SIGN     {sign_d:+.4f} CI [{sign_ci[0]:+.4f},{sign_ci[1]:+.4f}]")
    log(f"[cid] IDENTITY {iden_d:+.4f} CI [{iden_ci[0]:+.4f},{iden_ci[1]:+.4f}]")
    log(f"[cid] PASS={out['PASS']} -> {OUT} ({out['seconds']/60:.1f}m)")


if __name__ == "__main__":
    main()
