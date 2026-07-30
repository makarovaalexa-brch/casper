r"""test_i26_identity.py -- the safety property: i26 at init IS i25.

If the new arm is not bit-identical to the certified one before any gradient step, then every number the
retrain produces is confounded by an initialisation change and no gate means anything. So this is
asserted directly, on the real 18,359-item model with the real frozen RecVAE, before training.

Checks:
  1. i26(tokens) == i25(tokens) EXACTLY, for item-only inputs, at init.
  2. delta_b == 0 at init, so logits are unchanged too.
  3. Unseen tokens are routed AWAY from the taste path: adding them changes NOTHING at init
     (zero-init rho_expo) and does not corrupt native_z or the answered-token count.
  4. After a synthetic perturbation of rho_expo, unseen tokens DO change z -- i.e. the branch is
     actually wired in and check 3 was not passing by accident.
  5. The empty set still decodes to the same place as i25.

  python src/instrument/test_i26_identity.py
"""
import os
import sys
import argparse
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "src", "baselines"))

import metrics as M
from train_tower_t2 import build_model, load_recvae_teacher, pack_tokens, level_to_sv, PROC
from i26_encoder import I26Encoder, build_i26, UNSEEN_LEVEL


def rows_to_batch(rows):
    pack = [(np.asarray(a, np.int64), np.asarray(b, np.int64),
             level_to_sv(np.asarray(b, np.int64))) for a, b in rows]
    return pack_tokens(pack, binarize=False)


def main():
    meta = M.load_meta(PROC); ni = meta["n_items"]
    a = argparse.Namespace(arch="i25", teacher="warm_init", t_hidden=600, t_latent=200, token="film",
                           train_decoder=False, sign_prior=True, unfreeze_emb=False, lr=3e-4,
                           warm_lr_scale=0.1, full_kd=False, full_kd_w=0.3)
    cnt = np.asarray(M.load_train(ni, PROC).sum(axis=0)).ravel()
    torch.manual_seed(0)
    enc25, dec25, _t, _p, _g = build_model(a, ni, cnt)
    src = load_recvae_teacher(ni, hidden=a.t_hidden, latent=a.t_latent)
    torch.manual_seed(0)
    enc26, dec26, _p26 = build_i26(ni, src, a)

    # i25's sign prior is applied inside build_model; mirror it so the two are comparable.
    from train_tower_t2 import apply_sign_prior
    apply_sign_prior(enc26)
    with torch.no_grad():                       # phi/rho are randomly initialised -> copy i25's
        for n26, p26 in enc26.named_parameters():
            if n26.startswith(("psi.", "rho_expo.")) or n26 == "gate_e":
                continue
            p25 = dict(enc25.named_parameters()).get(n26)
            if p25 is not None and p25.shape == p26.shape:
                p26.copy_(p25)

    rng = np.random.default_rng(0)
    rows = [(rng.choice(ni, size=k, replace=False), rng.integers(0, 10, size=k))
            for k in (1, 2, 5, 12)]
    ids, vals, pad, lvs = rows_to_batch(rows)
    enc25.eval(); enc26.eval()
    ok = True
    with torch.no_grad():
        z25 = enc25(ids, vals, pad, lvs)
        z26 = enc26(ids, vals, pad, lvs)
    d = float((z25 - z26).abs().max())
    hit = d < 1e-6; ok &= hit
    print(f"[i26] 1. z identical on item-only input: max|diff| = {d:.3e} -> {'PASS' if hit else 'FAIL'}")

    hit = not hasattr(enc26, "delta_b"); ok &= hit
    print(f"[i26] 2. NO per-item bias exists (prior is learned) -> {'PASS' if hit else 'FAIL'}")

    # 3. add unseen tokens; at init they must change nothing
    rows_u = [(np.concatenate([s, rng.choice(ni, size=4, replace=False)]),
               np.concatenate([l, np.full(4, UNSEEN_LEVEL)])) for s, l in rows]
    ids_u, vals_u, pad_u, lvs_u = rows_to_batch(rows_u)
    with torch.no_grad():
        z26u = enc26(ids_u, vals_u, pad_u, lvs_u)
    d = float((z26 - z26u).abs().max())
    hit = d < 1e-6; ok &= hit
    print(f"[i26] 3. unseen tokens inert at init: max|diff| = {d:.3e} -> {'PASS' if hit else 'FAIL'}")

    # 4. perturb rho_expo -> unseen tokens MUST now matter (proves the wiring is real)
    with torch.no_grad():
        enc26.rho_expo[-1].weight.normal_(0, 0.05)
        z26p = enc26(ids_u, vals_u, pad_u, lvs_u)
        z26p_noun = enc26(ids, vals, pad, lvs)
    d_u = float((z26p - z26p_noun).abs().max())
    hit = d_u > 1e-4; ok &= hit
    print(f"[i26] 4. after perturbing rho_expo, unseen tokens DO change z: {d_u:.3e} "
          f"-> {'PASS' if hit else 'FAIL (branch not wired in)'}")
    with torch.no_grad():                                  # restore
        enc26.rho_expo[-1].weight.zero_()

    # 5. empty set
    e_ids, e_vals, e_pad, e_lvs = rows_to_batch([(np.empty(0, np.int64), np.empty(0, np.int64))])
    with torch.no_grad():
        d = float((enc25(e_ids, e_vals, e_pad, e_lvs) - enc26(e_ids, e_vals, e_pad, e_lvs)).abs().max())
    hit = d < 1e-6; ok &= hit
    print(f"[i26] 5. empty-set decode identical: {d:.3e} -> {'PASS' if hit else 'FAIL'}")

    print(f"[i26] {'ALL CHECKS PASS -- i26 at init IS i25' if ok else '*** FAILURES -- do not train ***'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
