r"""concept_train_v2.py -- train the concept channel on the FRESH v2 curriculum, pure NLL.

Differences from concept_distill_train.py, and only these:
  1. FRESH DRAWS. Examples come from concept_curriculum_v2.draw() on a live RNG, a new one every step.
     The old trainer materialised ONE example per user under a fixed seed, cached its latents, and looped
     that frozen set every epoch -- one draw per user for the whole run. That is the concept-side version
     of what limited the item tower, where even nine draws per user proved thin.
  2. NO DISTILLATION. lam>0 never beat lam=0 at equal budget, so the KL term and its teacher fold are
     gone. That is what makes (1) affordable: with no teacher there is nothing to precompute, and the
     base fold the student needs anyway is batched through fold_items_frozen.
  3. REFUSALS ARE FED, and the net is built with use_refusal=True so a refusal is distinguishable from a
     genuine MEH -- both carry v=0.0, so without the flag the model could not learn the distinction the
     interview regime exists to teach.

The frozen tower, the signed four-band answer, member-drop, the answerability rule and the target
hold-out are all unchanged.

  python src/instrument/concept_train_v2.py --snapshot models/i26c_ep31/t2i26c_ep31.pt \
         --tag cv2 --epochs 12 --steps_per_epoch 400 --batch 256
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "src", "baselines"))

import concept_curriculum_v2 as CV2
from concept_fold import ConceptFoldNet, fold_items_frozen
import concept_distill_train as CD

CKPT_DIR = os.path.join(_ROOT, ".cache", "instrument")
OUT = os.path.join(_ROOT, "experiments", "instrument")


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)


def _member_counts(S):
    """Per-concept member count, the weight the rare-concept sampler uses."""
    return np.array([len(S["member_sets"][i]) for i in range(len(S["tags"]))], np.float64)


def run_epoch(net, opt, S, args, ep, mc):
    Wd, bd, ni = S["Wd"], S["bd"], S["ni"]
    W0, b0 = Wd.clone(), bd.clone()
    users = S["users"]
    # Seed is a pure function of (run seed, epoch) and is LOGGED: fresh draws without a recorded seed
    # would be unreproducible.
    seed = args.seed * 100003 + ep
    rng = np.random.default_rng(seed)
    net.train()
    tot_nll, nb, t0 = 0.0, 0, time.time()
    regimes = {}
    n_ref_tok = 0
    n_tok = 0
    for step in range(args.steps_per_epoch):
        batch = []
        guard = 0
        while len(batch) < args.batch and guard < args.batch * 50:
            guard += 1
            u = users[int(rng.integers(len(users)))]
            ex = CV2.draw(u, rng, S["Mm"], S["grate"], S["member_sets"], S["item_mean"],
                          S["prereg"], member_count=mc)
            if ex is not None:
                batch.append(ex)
        if len(batch) < 2:
            continue
        seqs = [(e[0], e[1]) for e in batch]
        Zi = fold_items_frozen(S["enc"], seqs, batch=args.batch)      # batched frozen folds
        cids = [e[2] for e in batch]
        cvals = [e[3] for e in batch]
        crefs = [e[4] for e in batch]
        zout = net(Zi, cids, cvals, crefs)
        stu = zout @ Wd.T + bd
        logsm = F.log_softmax(stu, dim=-1)
        tgt = torch.zeros(len(batch), ni)
        for r, e in enumerate(batch):
            tgt[r, e[5]] = 1.0
        nll = -((logsm * tgt).sum(-1) / tgt.sum(-1).clamp_min(1.0)).mean()
        opt.zero_grad(); nll.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        tot_nll += float(nll); nb += 1
        for e in batch:
            regimes[e[6]] = regimes.get(e[6], 0) + 1
            n_ref_tok += e[7]; n_tok += len(e[2])
        if nb % 100 == 0:
            log(f"  ep{ep} step {nb}/{args.steps_per_epoch} NLL={tot_nll/nb:.4f} "
                f"({(time.time()-t0)/60:.1f}m)")
    assert torch.equal(Wd, W0) and torch.equal(bd, b0), "frozen decoder drifted"
    mix = {k: round(v / max(sum(regimes.values()), 1), 3) for k, v in sorted(regimes.items())}
    return {"nll": tot_nll / max(nb, 1), "minutes": (time.time() - t0) / 60, "seed": seed,
            "regime_mix": mix, "refusal_token_share": round(n_ref_tok / max(n_tok, 1), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=os.path.join(_ROOT, "models", "i26c_ep31", "t2i26c_ep31.pt"))
    ap.add_argument("--tag", default="cv2")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--steps_per_epoch", type=int, default=400)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--seed", type=int, default=98765)
    ap.add_argument("--max_users", type=int, default=0)
    ap.add_argument("--m_max", type=int, default=16)
    ap.add_argument("--mix", default=None, help="p_broad,p_dense,p_interview,p_kc0,p_kc1")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    if a.mix:
        CV2.set_mixture(*[float(x) for x in a.mix.split(",")])
    log(f"[cv2] mixture broad={CV2.P_BROAD} dense={CV2.P_DENSE} interview={CV2.P_INTERVIEW} "
        f"kc0={CV2.P_KC0} kc1={CV2.P_KC1} | rare pow={CV2.RARE_POW} clip={CV2.RARE_CLIP} "
        f"broad_cap={CV2.BROAD_CAP}")

    # reuse the existing stack builder verbatim (tower, tags, member sets, SEL valuation, users)
    a.lam = 0.0; a.arm = "s1"; a.tau = 1.0; a.eps_l2 = 0.0
    S = CD._build_train_stack(a)
    mc = _member_counts(S)
    log(f"[cv2] member counts: min {mc.min():.0f} median {np.median(mc):.0f} max {mc.max():.0f}")

    net = ConceptFoldNet(len(S["tags"]), d=S["enc"].d_out, h=a.hidden,
                         conc_init=S["d_c"].numpy(), use_refusal=True)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    log(f"[cv2] net params {sum(p.numel() for p in net.parameters()):,} (use_refusal=True)")

    if a.smoke:
        a.epochs, a.steps_per_epoch = 1, 3
    hist = []
    for ep in range(1, a.epochs + 1):
        r = run_epoch(net, opt, S, a, ep, mc)
        hist.append(r)
        log(f"[cv2] ep{ep} NLL={r['nll']:.4f} {r['minutes']:.1f}m seed={r['seed']} "
            f"mix={r['regime_mix']} refusal_tokens={r['refusal_token_share']:.3f}")
        torch.save({"net": net.state_dict(), "epoch": ep, "nll": r["nll"], "hist": hist,
                    "use_refusal": True, "snapshot": os.path.basename(a.snapshot)},
                   os.path.join(CKPT_DIR, f"{a.tag}_ep{ep:02d}.pt"))
    os.makedirs(OUT, exist_ok=True)
    json.dump(hist, open(os.path.join(OUT, f"{a.tag}_hist.json"), "w"), indent=2)
    log(f"[cv2] done -> {a.tag}")


if __name__ == "__main__":
    main()
