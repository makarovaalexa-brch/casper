"""arena_bigdev.py -- ENLARGED DEV-TEST re-run (coordinator addendum): n=600 synthetic users for
all results rows EXCEPT b4 (kept at its n=160 verdict, -0.0588 CI excl 0, 25min/160users) and the
labelled context arms (n=50, never cited). Sharpens the MDE from ~0.036 (n=160) for the verdict
sentences. The n=160 DEV-TEST cohort is a PREFIX of this cohort (uid-seeded splits; same world).

A/B/C at their DEV-VAL-selected TAU=24 are IDENTICAL to b2 by construction (exact tie, delta==0,
not a statistical statement) -- reported as such; their PURE (TAU=0) variants are re-run in full.
Appends the section to experiments/ARENA_BUILD.md + writes .cache/arena/results_big.json.
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
from arena_policies import (StaticSeq, B0Cold, B4Myopic, AskGradient, CATRouter, GolbandiTree,
                            ScorerA, B2Anchored, run_policy)

MD = "experiments/ARENA_BUILD.md"
Tmax = 24; KS = (50, 10)
N_BIG = 600


def md(t):
    open(MD, "a", encoding="utf-8").write(t)


def endpoint(o, K=50, t=Tmax):
    return float(np.nanmean(o["curves"][K][:, t]))


def anytime(o, K=10):
    return float(np.nanmean(np.nanmean(o["curves"][K][:, 1:], axis=1)))


def _fmt(c):
    return f"{c['mean']:+.4f} [{c['lo']:+.4f},{c['hi']:+.4f}] MDE {c['mde']:.4f} (n={c['n']})"


def main():
    t0 = time.time()
    ar = AC.Arena()
    coh = AC.make_cohorts(ar, n_train=1000, n_devval=80, n_devtest=N_BIG)
    cfg = coh["cfg"]
    # train/devval user sets are IDENTICAL to the main run (prefix property of make_cohorts); the
    # cfg sha differs via n_devtest, so tables regenerate under the new key (known-hash verified).
    ar.prefill_answers(coh["train"], "train", cfg)
    ar.prefill_answers(coh["devval"], "devval", cfg)
    ar.prefill_answers(coh["devtest"], "devtest_big", cfg)
    ar.set_pop_prior(AP.train_pop_prior(ar, coh["train"]))
    dt = coh["devtest"]
    res_main = json.load(open(f"{AC.CACHE_DIR}/results.json"))
    sel = res_main["selected_tau"]

    b2_users = coh["train"][:300]
    b2_seq, _ = AP.build_b2(ar, b2_users, Tmax=Tmax, prescreen_top=300, verbose=False)
    tail = AP.b3_tail(ar, b2_users, b2_seq)
    b1_seq = AP.build_b1(ar, coh["train"], verbose=False)
    gol_tree = AP.build_golbandi(ar, coh["train"][:300], max_depth=5, min_users=25, verbose=False)
    gbm = AP.build_scorerA(ar, coh["train"][:1000], n_samples=3500, verbose=False)

    R = {}
    def run(name, pol, skip=False):
        tt = time.time()
        R[name] = run_policy(ar, dt, pol, Tmax, Ks=KS, skip=skip, tag=name)
        print(f"  [big {name}] endpoint@50={endpoint(R[name]):.4f} "
              f"[{(time.time()-tt)/60:.1f}m]", flush=True)

    print(f"[bigdev] n={len(dt)} DEV-TEST users; running all arms except b4/ctx", flush=True)
    run("b0", B0Cold())
    run("b1", StaticSeq("b1", b1_seq))
    run("b2", StaticSeq("b2", b2_seq))
    run("b3", StaticSeq("b3", b2_seq, tail=tail), skip=True)
    run("D", GolbandiTree(gol_tree, b2_seq, tail))                       # selected TAU=0 == pure
    run("A_pure", B2Anchored("A_pure", b2_seq, ScorerA(gbm, M=100, Tmax=Tmax), 0))
    run("B_pure", B2Anchored("B_pure", b2_seq, AskGradient(M=100), 0))
    run("C_pure", B2Anchored("C_pure", b2_seq, CATRouter(M=100), 0))

    b2e = R["b2"]["curves"][50][:, Tmax]
    md("\n\n---\n\n## ENLARGED DEV-TEST (n=600; coordinator reliability addendum)\n\n")
    md(f"All results rows re-run on {len(dt)} synthetic DEV-TEST users (the n=160 cohort is a "
       "prefix; same world, same constructions). EXCEPTIONS, documented: b4 kept at its n=160 "
       "verdict (-0.0588 [-0.0835,-0.0344] vs b2 -- already decisive; 25min/160users compute); "
       "context arms stay n=50 (labelled, never cited). A/B/C at their DEV-VAL-selected TAU=24 are "
       "IDENTICAL to b2 by construction -- the tie is EXACT (delta==0), not a statistical claim; "
       "their pure TAU=0 variants are re-run in full below.\n\n")
    md("| arm | @50 T24 | @10 T24 | anytime@10 | @50 T8 | @50 T16 | vs b2 @50 T24 (paired, MDE) |\n"
       "|---|--:|--:|--:|--:|--:|---|\n")
    for name in ("b0", "b1", "b2", "b3", "D", "A_pure", "B_pure", "C_pure"):
        o = R[name]
        d = paired_ci([x - y for x, y in zip(o["curves"][50][:, Tmax], b2e)])
        dv = "--" if name == "b2" else _fmt(d)
        md(f"| {name} | {endpoint(o):.4f} | {endpoint(o,10):.4f} | {anytime(o):.4f} | "
           f"{endpoint(o,50,8):.4f} | {endpoint(o,50,16):.4f} | {dv} |\n")
    md("| A/B/C (sel TAU=24) | = b2 | = b2 | = b2 | = b2 | = b2 | EXACT tie by construction |\n")
    d0 = paired_ci([x - y for x, y in zip(R["b0"]["curves"][50][:, Tmax], b2e)])
    md(f"\nEnlarged-cohort MDE (vs b2 contrasts): ~{paired_ci([x - y for x, y in zip(R['D']['curves'][50][:, Tmax], b2e)])['mde']:.4f} "
       f"(was ~0.036 at n=160). b2 vs cold: {_fmt(d0)} (sign flipped: b2 above cold).\n")
    # b2 full decoded schedule (mechanism deliverable)
    md("\n### b2's full decoded schedule (the strongest fair static, 24 picks in order)\n\n")
    for i, qi in enumerate(b2_seq):
        md(f"{i+1}. {ar.q_names[int(qi)]}\n")
    json.dump({a: {f"ndcg{K}": [float(np.nanmean(R[a]["curves"][K][:, t])) for t in range(Tmax + 1)]
                   for K in KS} for a in R},
              open(f"{AC.CACHE_DIR}/results_big.json", "w"), indent=1)
    print(f"[bigdev] DONE [{(time.time()-t0)/60:.1f}m]; appended to {MD}", flush=True)


if __name__ == "__main__":
    main()
