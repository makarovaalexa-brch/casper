"""eval_ours_ml25m.py -- CANONICAL-SNAP check (read-only on checkpoints).

Scores our in-house recommender(s) through run_ml25m's EXACT evaluate path (run_ml25m.evaluate_ml25m,
which reuses signed_latent.build_splits / cohort / ndcg10 / SEEDS) and, for pbC, ALSO through the
canonical SIGNED eval (set_mn.eval_student) to separate scaffold identity from fold-in representation.

VERDICT question: do our models reproduce the recorded canonical
    pbC set-encoder   full/tail@10 = 0.4946 / 0.3372
    RecVAE-d512       full/tail@10 = 0.4998 / 0.3443
on run_ml25m's path?

NO training. NO data caps. Read-only on .cache checkpoints.
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import numpy as np
import torch
from scipy import sparse

torch.set_num_threads(4)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
sys.path.insert(0, os.path.join(_ROOT, "scripts", "instrument2"))

import signed_latent as SL
import run_ml25m as RM
import set_mn as SM

PBC = os.path.join(_ROOT, ".cache", "set_mn", "pbC_best.pt")
RECVAE_CKPT = SL.RECVAE_CKPT   # ".cache/instrument2/ml25m_recvae_d512_best.pt"


def load_pbC(ni):
    """Rebuild the pbC concept-capable set-encoder from checkpoint (belief pool, halfstar NLEV=15)."""
    blob = torch.load(PBC, map_location="cpu")
    sd = blob["student"]
    NT = sd["item_emb.weight"].shape[0]          # ni + NC
    pool = "belief" if any(k.startswith(("lam_head", "log_p0", "pscale")) for k in sd) else "attn"
    nlev = sd["gamma.weight"].shape[0]
    student = SM.SetEncoder(NT, token_mode="film", pool=pool, nlev=nlev, nknow=3)
    student.load_state_dict(sd)
    student.eval()
    decoder = torch.nn.Linear(SM.D, ni)
    decoder.load_state_dict(blob["decoder"])
    Wd = decoder.weight.detach(); bd = decoder.bias.detach()
    return student, Wd, bd, dict(NT=NT, pool=pool, nlev=nlev,
                                 rec_full=blob.get("full"), rec_tail=blob.get("tail"),
                                 epoch=blob.get("epoch"))


def pbC_predict_factory(student, Wd, bd, loved_level):
    """Adapter for run_ml25m.evaluate_ml25m: BINARY fold-in CSR -> pbC score.
    Each nonzero (liked) item is fed as a 'loved, know_well' answer (binary can't carry the star)."""
    def predict(X_csr):
        Xc = X_csr.tocsr()
        B = Xc.shape[0]
        rows = [Xc.indices[Xc.indptr[r]:Xc.indptr[r + 1]] for r in range(B)]
        L = max((len(x) for x in rows), default=1); L = max(L, 1)
        ids = np.zeros((B, L), np.int64); vals = np.zeros((B, L), np.float32)
        lvs = np.zeros((B, L), np.int64); pad = np.ones((B, L), bool)
        for r, cols in enumerate(rows):
            k = len(cols)
            if k:
                ids[r, :k] = cols; vals[r, :k] = 1.0; lvs[r, :k] = loved_level; pad[r, :k] = False
        with torch.no_grad():
            z = student(torch.from_numpy(ids), torch.from_numpy(vals),
                        torch.from_numpy(pad), torch.from_numpy(lvs))
            sc = (z @ Wd.T + bd).numpy().astype(np.float32)
        return sc
    return predict


def main():
    base = SL.load_arena_base(); ni = base["ni"]; headmask = base["headmask"]
    SL._POP = base["cnt"].astype(np.float64)
    print(f"[env] ni={ni} seeds={SL.SEEDS} cohort=te minus {len(base['study'])} study users", flush=True)

    # ============ pbC ============
    student, Wd, bd, meta = load_pbC(ni)
    print(f"[pbC] loaded {PBC}", flush=True)
    print(f"[pbC] NT={meta['NT']} pool={meta['pool']} nlev={meta['nlev']} "
          f"recorded full/tail={meta['rec_full']:.4f}/{meta['rec_tail']:.4f} (ep{meta['epoch']})", flush=True)

    # (A) CANONICAL signed eval (set_mn.eval_student) -- the path that produced 0.4946/0.3372.
    cf, ct = [], []
    for seed in SL.SEEDS:
        SPL = SL.build_splits(base, seed); users = SL.cohort(base, SPL, "test")
        f, t = SM.eval_student(student, None, base, SPL, users, Wd, bd)
        cf.append(f); ct.append(t)
        print(f"  [pbC canonical-signed] seed{seed} full={f:.4f} tail={t:.4f}", flush=True)
    print(f"[pbC canonical-signed] full={np.mean(cf):.4f} tail={np.mean(ct):.4f} "
          f"(target 0.4946/0.3372)", flush=True)

    # (B) run_ml25m EXACT evaluate path (evaluate_ml25m) with a BINARY-fold-in adapter.
    loved = meta["nlev"] - 1 if meta["nlev"] <= 5 else 9   # ordinal loved=nlev-1; halfstar loved=9
    predict = pbC_predict_factory(student, Wd, bd, loved)
    res = RM.evaluate_ml25m(predict, base, SL.SEEDS, headmask)
    print(f"[pbC run_ml25m-binary] full={res['full_ndcg@10']:.4f} tail={res['tail_ndcg@10']:.4f} "
          f"full@100={res['full_ndcg@100']:.4f}", flush=True)

    # ============ RecVAE-d512 ============
    if os.path.exists(RECVAE_CKPT):
        from recvae import RecVAE
        blob = torch.load(RECVAE_CKPT, map_location="cpu")
        a = blob["args"]; model = RecVAE(a["hidden"], a["latent"], ni)
        model.load_state_dict(blob["model"]); model.eval()

        def rv_predict(X_csr):
            Xc = X_csr.tocsr()
            with torch.no_grad():
                x = torch.tensor(Xc.toarray(), dtype=torch.float32)
                return model(x, calculate_loss=False).numpy().astype(np.float32)
        res_rv = RM.evaluate_ml25m(rv_predict, base, SL.SEEDS, headmask)
        print(f"[RecVAE run_ml25m-binary] full={res_rv['full_ndcg@10']:.4f} "
              f"tail={res_rv['tail_ndcg@10']:.4f} full@100={res_rv['full_ndcg@100']:.4f} "
              f"(target 0.4998/0.3443)", flush=True)
    else:
        print(f"[RecVAE] CHECKPOINT MISSING at {RECVAE_CKPT} -- weights not on disk; "
              f"cannot re-score. Recorded canonical = 0.4998/0.3443 "
              f"(ml25m_recvae_d512_TEST.json, same 5-seed protocol, MOSTPOP 0.2522).", flush=True)


if __name__ == "__main__":
    main()
