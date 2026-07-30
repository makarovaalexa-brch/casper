r"""run_arm_n.py -- arm N for every BINARY-NATIVE baseline (re-eval only, no retraining).

SPEC: docs/results/PROTOCOL_DISLIKE_DISCARD.md section 11.

These models' published input contract IS binary implicit feedback, so their arm-N input is
byte-identical to their arm-A input and NOTHING is retrained here. The only thing that varies is the
candidate pool. Every row therefore reports BOTH arms from the same fitted model, which makes the
per-model A->N delta an exact quantity rather than a difference of two runs, and re-checks the arm-A
snap on every pass -- if a row stops reproducing its canonical number, the harness changed, not the
protocol.

Ratings-native models (golbandi_native, RBMF, TaNP) are NOT here: they retrain on graded data and live
in their own scripts. `golbandi_node` and `rbmf_seed` DO appear here in their binary form, because the
binary-vs-native pair on the same pool is the cleanest statement of what the discard costs them.

  python src/baselines/run_arm_n.py [--only ease,recvae] [--max_minutes 600]
"""
import os
import sys
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M
import pop, itemknn, ease, ials, edlae, multvae, recvae, turbocf, golbandi_node, rbmf_seed, belief_mf
import sasrec, tanp_bestefffort as tanp
from arm_n import load_arm_n, _ROOT
from run_ml25m_liang import IALS_HP

OUTDIR = os.path.join(_ROOT, "experiments", "baselines", "arm_n")
CKPT = os.path.join(_ROOT, ".cache", "baselines")
CANON_DIR = os.path.join(_ROOT, "experiments", "baselines", "ml25m_liang")
ORDER = ["pop", "itemknn", "ease", "ials", "edlae", "turbocf", "golbandi_node", "rbmf_seed",
         "belief_mf", "recvae", "multvae", "tanp", "sasrec"]
TURBOCF_HP = {"alpha": 0.5, "power": 1.0, "filter": 2}   # val-selected, experiments/.../turbocf.json


def logln(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "run_arm_n.log"), "a") as f:
        f.write(line + "\n")


def canonical(name):
    """The recorded arm-A row, for the per-row snap check."""
    p = os.path.join(CANON_DIR, f"{name}.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return {"ndcg@10": d.get("ndcg@10"), "tail_ndcg@10": d.get("tail_ndcg@10")}


def build(name, D, max_minutes, log):
    """(predict_fn, hp). Identical construction to the canonical runners -- same hyperparameters, same
    checkpoints. Neural rows set epochs to the already-reached epoch so fit() restores best_state and
    trains nothing."""
    import argparse as ap
    train, ni = D["train"], D["n_items"]
    ck = os.path.join(CKPT, f"{name}_ml25m_liang.pt")

    if name == "pop":
        return pop.fit(train, ni, log=log), {}
    if name == "itemknn":
        pr = itemknn.fit(train, ni, args=ap.Namespace(topk=itemknn.DEFAULTS["topk"]), log=log)
        return pr, {"topk": pr.topk}
    if name == "ease":
        pr = ease.fit(train, ni, args=ap.Namespace(lam=ease.DEFAULTS["lam"]), log=log)
        return pr, {"lambda": pr.lam}
    if name == "ials":
        pr = ials.fit(train, ni, args=ap.Namespace(**IALS_HP), log=log)
        return pr, pr.hp
    if name == "edlae":
        # p was val-selected in arm A; reuse it rather than re-sweeping (the sweep is the record).
        p = json.load(open(os.path.join(CANON_DIR, "edlae.json")))["hp"]["p"]
        from ease import build_gram
        from edlae import edlae_B
        import gc
        G = build_gram(train); B = edlae_B(G, p); del G; gc.collect()
        return (lambda Xc, _B=B: np.asarray(Xc @ _B, np.float32)), {"p": p}
    if name == "turbocf":
        pr = turbocf.fit(train, ni, args=ap.Namespace(**TURBOCF_HP), log=log)
        return pr, TURBOCF_HP
    if name == "golbandi_node":
        pr = golbandi_node.fit(train, ni, args=ap.Namespace(**golbandi_node.DEFAULTS), log=log)
        return pr, {"k": pr.k, "weighted": pr.weighted}
    if name == "rbmf_seed":
        pr = rbmf_seed.fit(train, ni, log=log)
        return pr, getattr(pr, "hp", {})
    if name == "belief_mf":
        pr = belief_mf.fit(train, ni, log=log)
        return pr, getattr(pr, "hp", {})
    if name in ("recvae", "multvae", "tanp", "sasrec"):
        mod = {"recvae": recvae, "multvae": multvae, "tanp": tanp, "sasrec": sasrec}[name]
        a = mod._defaults("vae") if name == "multvae" else mod._defaults()
        if not os.path.exists(ck):
            raise SystemExit(f"[{name}] no checkpoint at {ck}; arm N re-evaluates, it does not train")
        import torch
        blob = torch.load(ck, map_location="cpu")
        a.epochs = int(blob["state"]["epoch"])       # loop range is empty -> restores best_state only
        a.max_minutes = 0.0
        pr = mod.fit(train, ni, evaluator=None, args=a, ckpt=ck, log=log)
        if name == "sasrec":
            pr.set_split("test")                      # per-user timestamp order: arm for the test pass
        return pr, {"best_val": getattr(pr, "best_val", None),
                    "best_epoch": getattr(pr, "best_epoch", None)}
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--max_minutes", type=float, default=1e9)
    a = ap.parse_args()
    order = ORDER if not a.only else [n for n in ORDER if n in set(a.only.split(","))]

    D = load_arm_n(log=logln)
    logln(f"=== arm N re-eval start; order={order} ===")
    for name in order:
        outp = os.path.join(OUTDIR, f"{name}.json")
        if os.path.exists(outp):
            logln(f"[skip] {name}: exists")
            continue
        logln(f"--- {name} ---")
        t0 = time.time()
        try:
            predict, hp = build(name, D, a.max_minutes, logln)
        except Exception as e:
            logln(f"[FAIL] {name}: {type(e).__name__}: {e}")
            continue
        arm_a = M.evaluate(predict, D["te_tr"], D["te_te"], batch_size=500, head_mask=D["head_mask"])
        arm_n = M.evaluate(predict, D["te_tr"], D["te_te"], batch_size=500, head_mask=D["head_mask"],
                           mask_X=D["pool"])
        can = canonical(name)
        snap = None
        if can and can["ndcg@10"] is not None:
            snap = abs(arm_a["ndcg@10"] - can["ndcg@10"]) <= 1e-3
            logln(f"[snap] {name}: arm A {arm_a['ndcg@10']:.4f} vs canonical {can['ndcg@10']:.4f} -> "
                  f"{'PASS' if snap else '*** FAIL ***'}")
        res = {"arm": "N", "model": name, "hp": hp, "A": arm_a, "N": arm_n,
               "snap_vs_canonical": snap, "seconds": time.time() - t0}
        json.dump(res, open(outp, "w"), indent=2)
        logln(f"[done] {name}: A full={arm_a['ndcg@10']:.4f} tail={arm_a['tail_ndcg@10']:.4f} | "
              f"N full={arm_n['ndcg@10']:.4f} tail={arm_n['tail_ndcg@10']:.4f} | "
              f"delta full={arm_n['ndcg@10'] - arm_a['ndcg@10']:+.4f} "
              f"tail={arm_n['tail_ndcg@10'] - arm_a['tail_ndcg@10']:+.4f} "
              f"({res['seconds'] / 60:.1f}m)")
    logln("=== arm N re-eval complete ===")


if __name__ == "__main__":
    main()
