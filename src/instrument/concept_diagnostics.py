r"""concept_diagnostics.py -- FREE structural diagnostics on a trained concept bolt-on (no training).

Three questions the cold-check curve cannot separate on its own. All run on an existing checkpoint.

D1 NORM-CAP BINDING. The fold ends with a per-user cap on ||z_out - z_items||. If that cap BINDS, it
   manufactures a ceiling at long budgets directly -- and it is a test-time constraint, so a binding cap
   can be raised and re-evaluated WITHOUT retraining. Reports the clipped fraction vs budget.

D2 GATE SELF-THROTTLING. The fold is recursive and the gate conditions on z_run, which already contains
   every prior shift. A known failure mode is per-step update magnitude collapsing with step index, so
   late answers barely land. Late-curve flatness (m=8->16 buys ~.0018 full) is consistent with BOTH gate
   saturation and information exhaustion, and those have opposite remedies. Reports ||g*delta|| by step.
   Steep decay (>3x from step 1 to 16) convicts the recursion and points at a set-symmetric fold.

D3 DECODER CEILING (oracle z). Gradient-descend z directly against each user's held-out likes through the
   FROZEN decoder. That is the best NDCG ANY fold could reach in this z-space, so it splits the
   concept-vs-item gap into "our fold is weak" vs "the channel's information is thin". If the ceiling is
   high, the decoder is exonerated and the gap belongs to input + fold.

  python src/instrument/concept_diagnostics.py --ckpt .cache/instrument/cv3_ep11.pt
"""
import os
import sys
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "src", "baselines"))

import concept_distill_train as CD
from concept_fold import ConceptFoldNet, fold_items_frozen
from concept_cold_check import build_banks, cold_interviews, ndcg_at, MS, log


def instrumented_forward(net, z_items, cids, cvals, crefs):
    """Replays ConceptFoldNet.forward, recording per-step ||g*delta|| and the final cap clip."""
    B = z_items.shape[0]
    z = z_items
    mmax = max((len(c) for c in cids), default=0)
    step_norms = []
    for step in range(mmax):
        has = torch.tensor([len(cids[r]) > step for r in range(B)])
        if not bool(has.any()):
            break
        cid = torch.tensor([cids[r][step] if len(cids[r]) > step else 0 for r in range(B)])
        val = torch.tensor([float(cvals[r][step]) if len(cvals[r]) > step else 0.0
                            for r in range(B)]).unsqueeze(1)
        p = net.emb(cid) * val
        if net.use_refusal:
            ref = torch.tensor([float(crefs[r][step]) if len(crefs[r]) > step else 0.0
                                for r in range(B)]).unsqueeze(1)
            x = torch.cat([z, p, val, ref], dim=1)
        else:
            x = torch.cat([z, p, val], dim=1)
        g = torch.sigmoid(net.gate(x))
        delta = net.mlp(x)
        upd = g * delta * has.float().unsqueeze(1)
        step_norms.append(float(upd[has].norm(dim=1).mean()))
        z = z + upd
    U = z - z_items
    cap = F.softplus(net.log_cap)
    nrm = U.norm(dim=1)
    clipped = float((nrm > cap).float().mean())
    return step_norms, clipped, float(cap), float(nrm.mean()), float(nrm.max())


def oracle_z(Wd, bd, targets, d, iters=300, lr=0.5):
    """Best z in this latent space for these targets, through the FROZEN decoder. Ceiling, not a model."""
    z = torch.zeros(1, d, requires_grad=True)
    opt = torch.optim.Adam([z], lr=lr)
    tg = torch.as_tensor(targets, dtype=torch.long)
    for _ in range(iters):
        opt.zero_grad()
        logp = F.log_softmax(z @ Wd.T + bd, dim=1)
        (-logp[0, tg].mean()).backward()
        opt.step()
    with torch.no_grad():
        return (z @ Wd.T + bd).numpy()[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(_ROOT, ".cache", "instrument", "cv3_ep11.pt"))
    ap.add_argument("--snapshot", default=os.path.join(_ROOT, "models", "i26c_ep31", "t2i26c_ep31.pt"))
    ap.add_argument("--n_users", type=int, default=1500)
    ap.add_argument("--n_oracle", type=int, default=150)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--max_users", type=int, default=0)
    a = ap.parse_args()
    a.lam = 0.0; a.arm = "s1"; a.tau = 1.0; a.eps_l2 = 0.0; a.m_max = 16

    S = CD._build_train_stack(a)
    mc = np.array([len(S["member_sets"][i]) for i in range(len(S["tags"]))], np.float64)
    from train_tower_t2 import compute_head_mask
    import metrics as M
    tr = M.load_train(S["ni"], os.path.join(_ROOT, "data", "ml-25m", "proc"))
    head_mask, _ = compute_head_mask(tr, S["ni"])

    blob = torch.load(a.ckpt, map_location="cpu")
    net = ConceptFoldNet(len(S["tags"]), d=S["enc"].d_out, h=a.hidden, conc_init=S["d_c"].numpy(),
                         use_refusal=blob.get("use_refusal", False))
    net.load_state_dict(blob["net"]); net.eval()
    log(f"net={os.path.basename(a.ckpt)} ep{blob['epoch']}  cap={float(F.softplus(net.log_cap)):.4f}")

    banks = build_banks(S)
    packs = cold_interviews(S, mc, n_users=a.n_users, rank=banks["conc_expinfo"])
    out = {}

    log("--- D1/D2: cap binding + per-step update norm (bank=expinfo)")
    with torch.no_grad():
        for m in (1, 2, 4, 8, 16):
            rows = packs[m][:512]
            Zi = fold_items_frozen(S["enc"], [(e[0], e[1]) for e in rows], batch=512)
            sn, clip, cap, nmean, nmax = instrumented_forward(
                net, Zi, [e[2] for e in rows], [e[3] for e in rows], [e[4] for e in rows])
            out[f"m{m}"] = {"clipped": clip, "cap": cap, "U_mean": nmean, "U_max": nmax,
                            "step_norms": sn}
            log(f"  m={m:2d}  clipped {clip:6.1%}  ||U|| mean {nmean:.3f} max {nmax:.3f} (cap {cap:.3f})")
            if m == 16:
                s1, s16 = sn[0], sn[-1]
                log(f"       per-step ||g*delta||: " + " ".join(f"{v:.3f}" for v in sn))
                log(f"       step1 {s1:.4f} -> step16 {s16:.4f}  decay {s1 / max(s16, 1e-9):.2f}x "
                    f"{'THROTTLING' if s1 / max(s16, 1e-9) > 3 else 'flat -- recursion OK'}")

    log(f"--- D3: oracle-z decoder ceiling on {a.n_oracle} users (frozen decoder, best possible z)")
    Wd, bd = S["Wd"], S["bd"]
    fs, ts = [], []
    for e in packs[0][:a.n_oracle]:
        sc = oracle_z(Wd, bd, e[5], S["enc"].d_out)
        f, t = ndcg_at(sc, e[5], head_mask)
        fs.append(f)
        if t is not None:
            ts.append(t)
    out["oracle_z"] = [float(np.mean(fs)), float(np.mean(ts))]
    log(f"  ORACLE-z ceiling  full {np.mean(fs):.4f}  tail {np.mean(ts):.4f}")
    log("  (concept m16 ~.1451/.0704, item m16 ~.2211/.0697, floor .1277/.0267)")

    json.dump(out, open(os.path.join(_ROOT, "experiments", "instrument",
                                     "concept_diagnostics.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
