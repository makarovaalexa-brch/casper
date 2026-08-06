r"""gtport_greedy_rescore.py -- score the GREEDY items-only order inside the gtport13 contract.

WHY. `GOLBANDI_PORT_ON_OUR_RULER.md` differences golbandi_tree against greedy items-only, but the
greedy rows come from `MASK_CONTRACT_DIAGNOSTIC.md`, a different harness
(`strategy_channel_suite.genv_eval`) with no per-user vectors on disk. A paired CI needs the SAME
users in the SAME order. Rather than admit a cross-run vector, we re-score the FIXED greedy order
through the identical path the gtport13 cells used: `adaptive_harness.run_interview` +
`run_interview_table.score_arm`, arm-N pool, NO-FLOOR, 10,000 test users, consumer `ours`.

CONTROL. `entropy0` is re-scored the same way and must reproduce the stored gtport13 per-user vector
EXACTLY (max abs per-user diff 0). If it does not, the greedy vectors are not in this contract and
must not be differenced.

The greedy order is read verbatim from `experiments/battery/greedy_static_i26c.json` (arm
"items-only"); nothing is re-selected. It is still a LOWER BOUND on a greedy search that had been
optimised for this pool, exactly as the diagnostic says.

  python src/baselines/gtport_greedy_rescore.py
"""
import os
import sys
import json
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "src", "instrument"))

import interview_strategies as ST                                          # noqa: E402
from arm_n import load_arm_n                                               # noqa: E402
from run_interview_table import build_recommenders, score_arm, mostpop_row, OUT  # noqa: E402
from adaptive_harness import run_interview, common_asked                   # noqa: E402
import golbandi_tree_selector as GT                                        # noqa: E402

LABEL = "gtport13_greedy"
LOGP = os.path.join(OUT, f"{LABEL}.log")
NPZ = os.path.join(OUT, "gtport13_greedy_per_user.npz")
SNAP = os.path.join(_ROOT, ".cache", "instrument", "t2i26c_ep31.pt")
BUDGETS = (1, 2, 4, 8)


def logln(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(LOGP, "a") as f:
        f.write(line + "\n")


def main():
    gj = json.load(open(os.path.join(_ROOT, "experiments", "battery", "greedy_static_i26c.json")))
    greedy = [int(i) for ch, i in gj["arms"]["items-only"]["sequence"] if int(ch) == 0]
    assert len(greedy) == 16, greedy
    logln(f"[greedy] items-only order (verbatim, greedy_static_i26c.json): {greedy}")

    D = load_arm_n(log=logln)
    ni, n = D["n_items"], D["te_tr"].shape[0]
    glob, bank = ST.build_orders(D["g_train"], D["train"], ni, log=logln)
    lvl = ST.level_lookup(D["g_te_tr"])
    built = build_recommenders(["ours"], D, arch="i26", snapshot=SNAP, log=logln)
    pop_row, prior = mostpop_row(D, log=logln)
    logln(f"[greedy] {n} users, {ni} items; mostpop prior {prior}")

    ORDERS = {"greedy-items-only": greedy, "entropy0-top16": list(glob["entropy0"][:16])}
    logln(f"[greedy] control order entropy0 top-16: {ORDERS['entropy0-top16']}")

    flat = {}
    if os.path.exists(NPZ):                                   # resume (HARD RULE 11)
        z = np.load(NPZ, allow_pickle=False)
        flat = {k: z[k] for k in z.files}
        logln(f"[greedy] resumed {len(flat)} vectors from {NPZ}")

    mode, obj = built["ours"]
    for name, order in ORDERS.items():
        for k in BUDGETS:
            if f"greedy_env|{name}|k{k}|full" in flat:
                continue
            sel = GT.StaticOrderTreeSelector(order, log=logln)
            asked, answered, _ = run_interview(sel, k, n, ni, lvl,
                                               ctx={"D": D, "glob": glob, "bank": bank},
                                               log=lambda *_: None)
            ga = common_asked(asked)
            assert ga == list(order[:k]), f"ask-list drift: {ga} vs {order[:k]}"
            t0 = time.time()
            c = score_arm("ours", mode, obj, asked, answered, D, pop_row, no_floor=True,
                          global_asked=ga, per_user=True)
            flat[f"greedy_env|{name}|k{k}|full"] = np.asarray(c["full_per_user"], np.float64)
            flat[f"greedy_env|{name}|k{k}|tail"] = np.asarray(c["tail_per_user"], np.float64)
            np.savez_compressed(NPZ, **flat)                  # write EVERY cell immediately
            logln(f"[greedy] {name:18s} k={k:2d} answered={c['answered']:.2f} "
                  f"full={c['full']:.4f} tail={c['tail']:.4f} "
                  f"(n_full={len(c['full_per_user'])} n_tail={len(c['tail_per_user'])}) "
                  f"{(time.time()-t0)/60:.2f}m -> {os.path.basename(NPZ)}")

    # ---------------------------------------------------------------- the admission control
    ref = np.load(os.path.join(OUT, "adaptive_table_gtport13_per_user.npz"), allow_pickle=False)
    worst = 0.0
    for k in BUDGETS:
        for ax in ("full", "tail"):
            d = np.abs(flat[f"greedy_env|entropy0-top16|k{k}|{ax}"]
                       - ref[f"ours|treestatic:entropy0|k{k}|{ax}"])
            worst = max(worst, float(d.max()))
            logln(f"[control] entropy0 k={k} {ax}: max per-user |re-score - gtport13| = {d.max():.3e}")
    logln(f"[control] {'PASS' if worst == 0.0 else 'CHECK'}: worst per-user diff {worst:.3e}")


if __name__ == "__main__":
    main()
