r"""arm_n_canary.py -- HARNESS GATE. Run this, and believe nothing from arm N until it passes.

SPEC: docs/results/PROTOCOL_DISLIKE_DISCARD.md section 11.2, rule 2.

WHAT IT TESTS. Most-Popular and EASE are binary-native: their arm-N input is BYTE-IDENTICAL to their
arm-A input. So the only thing that can move their numbers is the candidate pool. That makes them a
pure read on the harness:

  * both must RISE -- the pool now excludes each user's sub-3.5 ratings, which were unrankable junk
    sitting in everyone's candidate list.
  * the rise must be MODEST and of similar order for the two. A large or wildly unequal jump means the
    pool is eating something it should not (targets, or another user's row).
  * the pair must NOT REORDER. EASE is far above Most-Popular in arm A; a protocol change cannot
    invert that. If it does, the harness is broken, not the ranking.
  * arm A re-computed here must reproduce the recorded canonical rows EXACTLY (pop 0.1345/0.0226,
    ease 0.3476/0.2441). This is the snap that proves mask_X=None is a no-op refactor.

That last check is the one that matters most: it certifies that decoupling input from mask changed
nothing for the 40+ existing call sites.

  python src/baselines/arm_n_canary.py
"""
import os
import sys
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M
import pop
import ease
from arm_n import load_arm_n, _ROOT

OUTDIR = os.path.join(_ROOT, "experiments", "baselines", "arm_n")
# The recorded arm-A rows this refactor must reproduce bit-for-bit.
CANON = {"pop": {"ndcg@10": 0.1345, "tail_ndcg@10": 0.0226},
         "ease": {"ndcg@10": 0.3476, "tail_ndcg@10": 0.2441}}
SNAP_TOL = 5e-4


def logln(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "canary.log"), "a") as f:
        f.write(line + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="pop,ease")
    a = ap.parse_args()
    want = [s.strip() for s in a.only.split(",") if s.strip()]

    t0 = time.time()
    D = load_arm_n(log=logln)
    res = {}

    for name in want:
        ts = time.time()
        mod = {"pop": pop, "ease": ease}[name]
        predict = mod.fit(D["train"], D["n_items"], log=logln)
        # IDENTICAL input in both arms (both models are binary-native); only the pool differs.
        arm_a = M.evaluate(predict, D["te_tr"], D["te_te"], batch_size=500,
                           head_mask=D["head_mask"])
        arm_n = M.evaluate(predict, D["te_tr"], D["te_te"], batch_size=500,
                           head_mask=D["head_mask"], mask_X=D["pool"])
        res[name] = {"A": arm_a, "N": arm_n, "seconds": time.time() - ts}
        logln(f"[canary] {name:5s} A full={arm_a['ndcg@10']:.4f} tail={arm_a['tail_ndcg@10']:.4f} "
              f"| N full={arm_n['ndcg@10']:.4f} tail={arm_n['tail_ndcg@10']:.4f} "
              f"| delta full={arm_n['ndcg@10'] - arm_a['ndcg@10']:+.4f} "
              f"tail={arm_n['tail_ndcg@10'] - arm_a['tail_ndcg@10']:+.4f}")

    # ---------------------------------------------------------------- gates
    ok = True
    for name in want:
        for key, exp in CANON[name].items():
            got = res[name]["A"][key]
            hit = abs(got - exp) <= SNAP_TOL
            ok &= hit
            logln(f"[gate] SNAP  {name}.{key}: got {got:.4f} vs canonical {exp:.4f} "
                  f"-> {'PASS' if hit else 'FAIL'}")
    for name in want:
        d = res[name]["N"]["ndcg@10"] - res[name]["A"]["ndcg@10"]
        hit = d > 0
        ok &= hit
        logln(f"[gate] RISE  {name}: {d:+.4f} -> {'PASS' if hit else 'FAIL (pool must only remove junk)'}")
    if "pop" in res and "ease" in res:
        hit = res["ease"]["N"]["ndcg@10"] > res["pop"]["N"]["ndcg@10"]
        ok &= hit
        logln(f"[gate] ORDER ease > pop under N: {res['ease']['N']['ndcg@10']:.4f} vs "
              f"{res['pop']['N']['ndcg@10']:.4f} -> {'PASS' if hit else 'FAIL'}")
        dp = res["pop"]["N"]["ndcg@10"] - res["pop"]["A"]["ndcg@10"]
        de = res["ease"]["N"]["ndcg@10"] - res["ease"]["A"]["ndcg@10"]
        logln(f"[gate] EVEN  rise pop {dp:+.4f} vs ease {de:+.4f} (ratio "
              f"{(de / dp if dp else float('nan')):.2f}) -- INSPECT, not automatic")

    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "canary.json"), "w") as f:
        json.dump(res, f, indent=2)
    logln(f"[canary] {'ALL GATES PASS' if ok else '*** GATE FAILURE -- arm N is NOT usable ***'} "
          f"in {(time.time() - t0) / 60:.1f} m")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
