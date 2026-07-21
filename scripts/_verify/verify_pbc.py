"""verify_pbc.py -- forensic snap-check of set_mn checkpoints under their TRAINING-TIME (Jul-15) forward.

Each checkpoint is scored through the PINNED set_mn.py version that existed when it was trained
(SetEncoder forward + eval_student + sv_to_level from that commit), against the CANONICAL current
signed_latent data/eval primitives (load_arena_base/build_splits/cohort/ndcg10 -- the fixed ruler;
signed_latent.py has only ONE tracked version, created Jul-20 @7a12727, so it is the only baseline
available and is shared identically here as in the live scaffold).

Pinned set_mn shas (last commit <= each ckpt's mtime):
  pbC_best.pt   (mtime 2026-07-15 07:57)  -> 88777ed  (halfstar, NLEV=15, belief pool)
  paord_best.pt (mtime 2026-07-15 11:28)  -> 8a01921  (ordinal grading collapse, NLEV=5, belief pool)
  pb2_best.pt   (mtime 2026-07-14 05:02)  -> 2bace5e  (halfstar, NLEV=10, attn pool)

READ-ONLY on .cache. NO training. NO data caps (full 200-user test cohort x 5 seeds).
"""
import os, sys, importlib.util
os.environ.setdefault("OMP_NUM_THREADS", "4"); os.environ.setdefault("MKL_NUM_THREADS", "4")
import numpy as np, torch
torch.set_num_threads(4)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
sys.path.insert(0, os.path.join(_ROOT, "scripts", "instrument2"))
sys.path.insert(0, _HERE)

import signed_latent as SL   # current (only tracked) version: the fixed data/eval ruler

CACHE = os.path.join(_ROOT, ".cache", "set_mn")


def load_pinned(modname, fname):
    """Load a pinned set_mn_*.py as an isolated module (each defines SetEncoder/eval_student/sv_to_level)."""
    path = os.path.join(_HERE, fname)
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


def score(ckpt, SM, grading=None):
    blob = torch.load(os.path.join(CACHE, ckpt), map_location="cpu")
    sd = blob["student"]
    NT = sd["item_emb.weight"].shape[0]
    nlev = sd["gamma.weight"].shape[0]
    nknow = sd["know_emb.weight"].shape[0] if "know_emb.weight" in sd else 0
    pool = "belief" if any(k.startswith(("lam_head", "log_p0", "pscale")) for k in sd) else "attn"
    if grading is not None and hasattr(SM, "set_grading"):
        SM.set_grading(grading)     # sets module GRADING/NLEV so sv_to_level matches training
    student = SM.SetEncoder(NT, token_mode="film", pool=pool, nlev=nlev, nknow=nknow)
    student.load_state_dict(sd)
    student.eval()
    ni = blob["decoder"]["weight"].shape[0]
    decoder = torch.nn.Linear(SM.D, ni)
    decoder.load_state_dict(blob["decoder"])
    Wd = decoder.weight.detach(); bd = decoder.bias.detach()

    base = SL.load_arena_base()
    assert base["ni"] == ni, f"ni mismatch {base['ni']} vs decoder {ni}"
    SL._POP = base["cnt"].astype(np.float64)
    ff, tt = [], []
    for seed in SL.SEEDS:
        SPL = SL.build_splits(base, seed)
        users = SL.cohort(base, SPL, "test")
        f, t = SM.eval_student(student, None, base, SPL, users, Wd, bd)
        ff.append(f); tt.append(t)
        print(f"    seed{seed:>2}  full={f:.4f} tail={t:.4f}  (n={len(users)})", flush=True)
    mf, mt = float(np.mean(ff)), float(np.mean(tt))
    rf, rt = blob.get("full"), blob.get("tail")
    df, dt = mf - rf, mt - rt
    snap = abs(df) <= 0.003 and abs(dt) <= 0.003
    print(f"  [{ckpt}] pool={pool} nlev={nlev} nknow={nknow} grading={grading or 'halfstar(fixed)'}")
    print(f"  [{ckpt}] MEASURED full/tail = {mf:.4f}/{mt:.4f}   RECORDED = {rf:.4f}/{rt:.4f}"
          f"   delta = {df:+.4f}/{dt:+.4f}   -> {'SNAP' if snap else 'NO-SNAP'}", flush=True)
    return mf, mt, rf, rt, snap


def main():
    jobs = [
        ("pbC_best.pt",   "88777ed", "set_mn_jul15.py", None),
        ("paord_best.pt", "8a01921", "set_mn_paord.py", "ordinal"),
        ("pb2_best.pt",   "2bace5e", "set_mn_pb2.py",   None),
    ]
    results = []
    for ckpt, sha, fname, grading in jobs:
        print(f"\n==== {ckpt}  (pinned set_mn @ {sha})  grading={grading or 'halfstar'} ====", flush=True)
        SM = load_pinned(f"setmn_{sha}", fname)
        results.append((ckpt, sha) + score(ckpt, SM, grading))
    print("\n================ VERDICT TABLE ================")
    for ckpt, sha, mf, mt, rf, rt, snap in results:
        print(f"  {ckpt:16s} pin={sha}  measured {mf:.4f}/{mt:.4f}  recorded {rf:.4f}/{rt:.4f}  "
              f"{'SNAP (+/-0.003)' if snap else 'NO-SNAP'}")


if __name__ == "__main__":
    main()
