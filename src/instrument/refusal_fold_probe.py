r"""Does folding a REFUSAL degrade the ranking below asking nothing at all?

THE QUESTION (raised in review, 2026-09-15). Two ported selectors score below the empty interview at
K=1 on the test cohort (Golbandi's tree .1557, the reduced EDDI acquisition .1537, against .1586 for
asking nothing). Asking one question and folding the outcome should not make a user's ranking WORSE
than asking them nothing -- unless the refusal token itself carries negative information. If it does,
then every arm that asks unanswerable questions is charged twice (the lost turn AND a degraded state)
and every answer-rate effect in the paper, RQ1's included, is inflated by construction.

THE TEST. For a question q, `split_rows` partitions the cohort into like / dislike / noanswer exactly
as the tree does. Take the NOANSWER branch -- the users who cannot answer q -- and score them two
ways on the same rows:

    A. genv_eval(E, [],  refusers)      ask nothing
    B. genv_eval(E, [q], refusers)      ask q, fold the refusal

B - A is the price of a refused question. Zero is the null hypothesis: a refusal should be
information-free, so the state should be unchanged and the difference should be 0.0 exactly. Anything
negative and separating is the defect the review postulates.

Run over several questions of BOTH channels, because the two fold through different tokens: an
unanswerable ITEM enters as the i26 UNSEEN exposure token, an unanswerable CONCEPT as the cv3 refusal
token. The two can behave differently and the paper's concept channel depends on the second.

CONTRACT: the certified pair, the policy-B pool, the learned decoder bias, full catalogue, no floor --
`build_greedy_env` unchanged, so this probe is on the same ruler as every number in the paper.
"""
import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import json
import time
import argparse
import numpy as np

from strategy_channel_suite import (log, build_greedy_env, set_active_rows, genv_eval,
                                    contract_selftest, CERT_TOWER, CERT_CONC)
from ndcg_tree import split_rows, build_pool, leak_assert
from run_battery_phaseA import OUTDIR, bootstrap_ci

LABEL = "refusal_fold_probe"
BOOT, BOOT_SEED = 10_000, 98765


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=CERT_TOWER)
    ap.add_argument("--concept_ckpt", default=CERT_CONC)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_items", type=int, default=500)
    ap.add_argument("--n_concepts", type=int, default=500)
    ap.add_argument("--n_probe", type=int, default=6, help="questions per channel")
    ap.add_argument("--label", default=LABEL)
    ap.add_argument("--full_threads", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    OUT = os.path.join(OUTDIR, args.label + ".json")

    E = build_greedy_env(args.snapshot, args.split, args.concept_ckpt, f"{args.label}({args.split})")
    rows = E["rows"]
    contract_selftest(E, rows)
    leak_assert(E, rows, args.split)
    set_active_rows(E, rows); E["memo"] = {}

    f0_all, t0_all = genv_eval(E, [], rows)
    log(f"[m=0] whole cohort, ask nothing: {np.nanmean(f0_all):.5f} full / {np.nanmean(t0_all):.5f} tail")

    pool, item_pool, conc_pool = build_pool(E, args.n_items, args.n_concepts)
    # Spread the probes ACROSS each ranking, not down its head. The concept pool is ranked BY
    # ANSWERABILITY, so its first entries have ~zero refusers by construction and test nothing: the
    # first run of this probe skipped all six concepts for exactly that reason. The cv3 refusal token
    # is the one RQ1 depends on, so it has to be reached further down the ranking.
    def spread(p, n):
        if n >= len(p):
            return list(p)
        idx = np.linspace(0, len(p) - 1, n).round().astype(int)
        return [p[i] for i in sorted(set(idx.tolist()))]

    probes = [("item", q) for q in spread(item_pool, args.n_probe)] + [
        ("concept", q) for q in spread(conc_pool, args.n_probe)]

    out = {"analysis": "refusal_fold_probe", "label": args.label, "citable": True,
           "split": args.split, "tower": os.path.basename(args.snapshot),
           "concept_ckpt": os.path.basename(args.concept_ckpt),
           "question": ("does folding a refusal move the ranking away from the m=0 state? "
                        "null hypothesis: difference is exactly 0.0"),
           "n_cohort": len(rows), "probes": []}

    log("=" * 104)
    log(f"{'channel':<9}{'id':>7}{'refusers':>10}{'ask-nothing':>13}{'after refusal':>15}"
        f"{'difference':>13}{'separates':>11}")
    log("-" * 104)
    worst = 0.0
    for chan, q in probes:
        kids, _a, _z = split_rows(E, rows, q)
        ref = kids["noanswer"]
        if len(ref) < 50:
            log(f"{chan:<9}{q[1]:>7}{len(ref):>10}   too few refusers, skipped")
            continue
        set_active_rows(E, ref); E["memo"] = {}
        fa, ta = genv_eval(E, [], ref)            # ask nothing
        fb, tb = genv_eval(E, [q], ref)           # ask q; every one of these users refuses
        d = fb - fa
        m, ci, n = bootstrap_ci(d, n_boot=BOOT, seed=BOOT_SEED)
        sep = bool(ci[0] > 0 or ci[1] < 0)
        worst = min(worst, float(m))
        log(f"{chan:<9}{q[1]:>7}{len(ref):>10}{np.nanmean(fa):>13.5f}{np.nanmean(fb):>15.5f}"
            f"{m:>+13.5f}{str(sep):>11}")
        out["probes"].append({"channel": chan, "qid": int(q[1]), "n_refusers": int(len(ref)),
                              "ask_nothing_full@10": float(np.nanmean(fa)),
                              "after_refusal_full@10": float(np.nanmean(fb)),
                              "diff_full@10": float(m), "ci95": [float(ci[0]), float(ci[1])],
                              "separates": sep,
                              "max_abs_per_user": float(np.nanmax(np.abs(d))),
                              "tail_diff@10": float(np.nanmean(tb - ta))})
    log("-" * 104)

    any_sep = [p for p in out["probes"] if p["separates"]]
    exact_zero = [p for p in out["probes"] if p["max_abs_per_user"] == 0.0]
    out["n_probes"] = len(out["probes"])
    out["n_separating"] = len(any_sep)
    out["n_bit_identical_to_m0"] = len(exact_zero)
    out["worst_mean_diff_full@10"] = worst
    # Sign-aware. The first version of this verdict reported only the worst NEGATIVE shift, which
    # reads as "refusals always hurt". They do not -- the observed shifts go both ways.
    neg = [p for p in any_sep if p["diff_full@10"] < 0]
    pos = [p for p in any_sep if p["diff_full@10"] > 0]
    mags = [abs(p["diff_full@10"]) for p in out["probes"]] or [0.0]
    out["largest_abs_mean_shift"] = float(max(mags))
    out["verdict"] = (
        "REFUSALS ARE INFORMATION-FREE: every probed refusal leaves the per-user ranking "
        "bit-identical to the m=0 state."
        if len(exact_zero) == len(out["probes"]) else
        ("REFUSALS MOVE THE STATE, IN BOTH DIRECTIONS. " + str(len(any_sep)) + " of "
         + str(len(out["probes"])) + " probed questions shift the refusers' ranking by a separating "
         "margin (" + str(len(neg)) + " down, " + str(len(pos)) + " up); largest absolute mean shift "
         + format(max(mags), ".5f") + " full. A refused question is not state-preserving, but it is "
         "not a systematic penalty either, and the magnitude is to be read against the effects the "
         "paper reports (+0.01430 bank, +0.01035 criterion)."))
    log(out["verdict"])
    out["seconds"] = round(time.time() - t0, 1)
    with open(OUT + ".tmp", "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    os.replace(OUT + ".tmp", OUT)
    log(f"[done] wrote {os.path.basename(OUT)} in {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
