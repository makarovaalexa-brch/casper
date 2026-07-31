r"""select_i26c.py -- PRE-REGISTERED checkpoint selection for the i26 certification run.

WHY THIS IS A SCRIPT AND NOT A JUDGEMENT CALL AT 6 A.M.
Run 1's rule was argmax(sel). It picked ep18. ep20 then beat ep18 on TEST on BOTH axes (full 0.3416 vs
0.3386, interview-at-8 0.2078 vs 0.2014), so the rule was mis-specified -- and re-picking ep20 after
seeing those TEST numbers would have been selection on test. The rule below is fixed in advance, reads
ONLY val quantities that the trainer already wrote into each checkpoint, and never opens the test set.

THE RULE:  score = sel + val_full,  maximised.
  sel      = 0.5 * (val NDCG@10 at a 2-question interview + at an 8-question interview)
  val_full = val NDCG@10 on the full profile
Equal weight to interview capability and full-profile capability -- which is exactly the "balanced
instrument" the chapter claims. Sanity check on run 1: ep20 .4801 > ep14 .4791 > ep18 .4780, i.e. it
selects ep20 from val alone, which is the checkpoint that did in fact win on test.

Also prints the val PARETO FRONT (no other checkpoint better on both axes), because the author may want
to override the scalarisation with the tradeoff curve in front of them. Selection is a judgement the
author is entitled to make; this script makes the DEFAULT explicit and reproducible.

  python src/instrument/select_i26c.py [--tag t2i26c] [--dir .cache/instrument] [--copy]
"""
import os
import sys
import glob
import json
import argparse
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))


def load_rows(ckpt_dir, tag):
    rows = []
    for p in sorted(glob.glob(os.path.join(ckpt_dir, f"{tag}_ep*.pt"))):
        if ".archived_" in p or ".tmp" in p:
            continue
        try:
            b = torch.load(p, map_location="cpu")
        except Exception as e:                      # a half-written file must not abort the selection
            print(f"  !! UNREADABLE {os.path.basename(p)}: {e}")
            continue
        if b.get("sel") is None or b.get("val_full") is None:
            print(f"  !! SKIP {os.path.basename(p)}: missing sel/val_full")
            continue
        rows.append({"path": p, "epoch": int(b["epoch"]), "sel": float(b["sel"]),
                     "val_full": float(b["val_full"]), "val_tail": float(b.get("val_tail", float("nan"))),
                     "k2": float(b.get("k2", float("nan"))), "k8": float(b.get("k8", float("nan"))),
                     "k0": float(b.get("k0", float("nan"))), "k1": float(b.get("k1", float("nan")))})
    return sorted(rows, key=lambda r: r["epoch"])


def pareto(rows):
    """Val-Pareto front on (sel, val_full): keep r if nothing else is >= on both and > on one."""
    front = []
    for r in rows:
        dominated = any((o["sel"] >= r["sel"] and o["val_full"] >= r["val_full"] and
                         (o["sel"] > r["sel"] or o["val_full"] > r["val_full"])) for o in rows)
        if not dominated:
            front.append(r)
    return front


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="t2i26c")
    ap.add_argument("--dir", default=os.path.join(_ROOT, ".cache", "instrument"))
    ap.add_argument("--copy", action="store_true",
                    help="copy the winner to <tag>_SELECTED.pt (a copy, never a move -- the epoch file "
                         "stays exactly where it is)")
    a = ap.parse_args()

    rows = load_rows(a.dir, a.tag)
    if not rows:
        print(f"NO CHECKPOINTS matching {a.tag}_ep*.pt in {a.dir}")
        sys.exit(1)

    print(f"=== {len(rows)} checkpoints, tag={a.tag} ===")
    print(f"{'ep':>3} {'sel':>8} {'val_full':>9} {'val_tail':>9} {'k2':>8} {'k8':>8} "
          f"{'k0':>8} {'k1':>8} {'SCORE':>8}")
    for r in rows:
        r["score"] = r["sel"] + r["val_full"]
    for r in rows:
        print(f"{r['epoch']:>3} {r['sel']:>8.4f} {r['val_full']:>9.4f} {r['val_tail']:>9.4f} "
              f"{r['k2']:>8.4f} {r['k8']:>8.4f} {r['k0']:>8.4f} {r['k1']:>8.4f} {r['score']:>8.4f}")

    front = sorted(pareto(rows), key=lambda r: -r["sel"])
    print("\n=== VAL PARETO FRONT (sel vs val_full) -- the author may override on this curve ===")
    for r in front:
        print(f"  ep{r['epoch']:02d}  sel={r['sel']:.4f}  val_full={r['val_full']:.4f}  "
              f"score={r['score']:.4f}")

    win = max(rows, key=lambda r: r["score"])
    print(f"\n=== SELECTED (pre-registered rule: max sel + val_full) ===")
    print(f"  epoch {win['epoch']}  sel={win['sel']:.4f}  val_full={win['val_full']:.4f}  "
          f"score={win['score']:.4f}")
    print(f"  {win['path']}")

    # Context the author asked to see next to any pick. VAL numbers only -- the test comparison is a
    # separate, later step, and must not influence the choice above.
    print("\n  references (val):  certified i25 val_full = 0.3451 | RecVAE own val = 0.3512")
    print("                     step-0 of THIS run (untrained, RecVAE init) = 0.3510")

    out = {"tag": a.tag, "rule": "argmax(sel + val_full)", "selected_epoch": win["epoch"],
           "selected_path": win["path"], "score": win["score"], "sel": win["sel"],
           "val_full": win["val_full"], "val_tail": win["val_tail"],
           "pareto_front": [r["epoch"] for r in front], "n_checkpoints": len(rows),
           "all": [{k: v for k, v in r.items() if k != "path"} for r in rows]}
    jp = os.path.join(_ROOT, "experiments", "instrument", f"{a.tag}_selection.json")
    json.dump(out, open(jp, "w"), indent=2)
    print(f"\n  wrote {jp}")

    if a.copy:
        import shutil
        dst = os.path.join(a.dir, f"{a.tag}_SELECTED.pt")
        shutil.copy2(win["path"], dst)           # COPY: the epoch file is never moved or removed
        print(f"  copied -> {dst}")


if __name__ == "__main__":
    main()
