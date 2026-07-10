"""arena_insample.py -- AUTHOR DIAGNOSTIC: in-sample evaluation of b2 + PURE adaptive classes on
~200 users FROM THEIR OWN TRAINING COHORTS. Splits the DEV tie into its two diseases:
  (i) win in-sample, tie held-out  -> GENERALIZATION GAP (data/regularization fixable);
  (ii) cannot win even in-sample   -> EXPRESSIVENESS/OPTIMIZATION failure (scaling will not save it).
D (tree + b2 tail) is superset-of-static by construction: its in-sample number sanity-checks the
harness (must be >= b2 in-sample, else something is miswired).
B/C are calibration-only (no training cohort): their 'in-sample' equals held-out by construction --
included for completeness, labelled.
Appends the table + one-line disease verdicts to experiments/ARENA_BUILD.md.
"""
import os, sys, json, time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.join(_HERE, "instrument2"))
import warnings
warnings.filterwarnings("ignore")

import arena_core as AC
import arena_policies as AP
from arena_core import paired_ci
from arena_policies import (StaticSeq, AskGradient, CATRouter, GolbandiTree, ScorerA, B2Anchored,
                            run_policy)

MD = "experiments/ARENA_BUILD.md"
Tmax = 24; KS = (50, 10)
N_IN = 200


def md(t):
    open(MD, "a", encoding="utf-8").write(t)


def endpoint(o, K=50, t=Tmax):
    return float(np.nanmean(o["curves"][K][:, t]))


def _fmt(c):
    return f"{c['mean']:+.4f} [{c['lo']:+.4f},{c['hi']:+.4f}] MDE {c['mde']:.4f} (n={c['n']})"


def main():
    t0 = time.time()
    ar = AC.Arena()
    coh = AC.make_cohorts(ar, n_train=1000, n_devval=80, n_devtest=160)
    cfg = coh["cfg"]
    ar.prefill_answers(coh["train"], "train", cfg, verbose=False)
    ar.set_pop_prior(AP.train_pop_prior(ar, coh["train"]))
    ins = coh["train"][:N_IN]      # subset of b2's (train[:300]), D's (train[:300]), A's (train[:1000])

    b2_seq, _ = AP.build_b2(ar, coh["train"][:300], Tmax=Tmax, prescreen_top=300, verbose=False)
    tail = AP.b3_tail(ar, coh["train"][:300], b2_seq)
    gol_tree = AP.build_golbandi(ar, coh["train"][:300], max_depth=5, min_users=25, verbose=False)
    gbm = AP.build_scorerA(ar, coh["train"][:1000], n_samples=3500, verbose=False)

    # DEV-TEST (held-out) references from the main run
    dev = json.load(open(f"{AC.CACHE_DIR}/results.json"))["curves_mean"]
    dev_ep = {a: dev[a]["ndcg50"][Tmax] for a in ("b2", "A_pure", "B_pure", "C_pure", "D_pure")}

    R = {}
    def run(name, pol):
        tt = time.time()
        R[name] = run_policy(ar, ins, pol, Tmax, Ks=KS, tag=name)
        print(f"  [in-sample {name}] endpoint@50={endpoint(R[name]):.4f} "
              f"[{(time.time()-tt)/60:.1f}m]", flush=True)

    print(f"[insample] {N_IN} users from the construction cohorts", flush=True)
    run("b2", StaticSeq("b2", b2_seq))
    run("A_pure", B2Anchored("A_pure", b2_seq, ScorerA(gbm, M=100, Tmax=Tmax), 0))
    run("B_pure", B2Anchored("B_pure", b2_seq, AskGradient(M=100), 0))
    run("C_pure", B2Anchored("C_pure", b2_seq, CATRouter(M=100), 0))
    run("D_pure", GolbandiTree(gol_tree, b2_seq, tail))

    b2e = R["b2"]["curves"][50][:, Tmax]
    md("\n\n---\n\n## IN-SAMPLE DIAGNOSTIC (author-directed): which disease is the DEV tie?\n\n")
    md(f"b2 + PURE (tau=0) adaptive classes evaluated on {N_IN} users FROM THEIR OWN TRAINING "
       "cohorts (b2/D constructed on train[:300]; A labelled on train[:1000]; these 200 are a "
       "subset of both). Interpretation rule (pre-stated): win in-sample but tie/lose held-out = "
       "GENERALIZATION GAP (data/regularization fixable); cannot win even in-sample = "
       "EXPRESSIVENESS/OPTIMIZATION failure (scaling will not save it). B/C are calibration-only "
       "(no training cohort): in-sample == held-out by construction, rows labelled accordingly. "
       "D (tree + b2 tail) is superset-of-static by construction: its in-sample number must be "
       ">= b2's or the harness is miswired (sanity check).\n\n")
    md("| arm | in-sample @50 T24 | vs b2 in-sample (paired) | DEV-TEST @50 T24 (n=160) | "
       "in-sample - DEV gap |\n|---|--:|---|--:|--:|\n")
    verdicts = []
    for name in ("b2", "A_pure", "B_pure", "C_pure", "D_pure"):
        o = R[name]
        e = endpoint(o)
        dvs = "--" if name == "b2" else _fmt(paired_ci(
            [x - y for x, y in zip(o["curves"][50][:, Tmax], b2e)]))
        gap = e - dev_ep[name]
        md(f"| {name} | {e:.4f} | {dvs} | {dev_ep[name]:.4f} | {gap:+.4f} |\n")
        if name != "b2":
            d = paired_ci([x - y for x, y in zip(o["curves"][50][:, Tmax], b2e)])
            if d["lo"] > 0:
                verdicts.append((name, "wins IN-SAMPLE -> the DEV tie is a GENERALIZATION GAP "
                                       "(data/regularization fixable)"))
            elif d["hi"] < 0:
                verdicts.append((name, "LOSES even in-sample -> EXPRESSIVENESS/OPTIMIZATION "
                                       "failure (scaling will not save this class)"))
            else:
                verdicts.append((name, "TIES even in-sample -> EXPRESSIVENESS/OPTIMIZATION "
                                       "failure (cannot exploit its own training users)"))
    md("\n**Disease verdicts (one line per class):**\n")
    for n, v in verdicts:
        md(f"- {n}: {v}\n")
    dd = paired_ci([x - y for x, y in zip(R["D_pure"]["curves"][50][:, Tmax], b2e)])
    md(f"- harness sanity (D superset-of-static): D in-sample vs b2 = {_fmt(dd)}; "
       f"{'OK (>= b2 within CI)' if dd['hi'] >= 0 else 'FLAG: D BELOW b2 in-sample -- investigate'}\n")
    json.dump({a: endpoint(R[a]) for a in R}, open(f"{AC.CACHE_DIR}/results_insample.json", "w"),
              indent=1)
    print(f"[insample] DONE [{(time.time()-t0)/60:.1f}m]", flush=True)


if __name__ == "__main__":
    main()
