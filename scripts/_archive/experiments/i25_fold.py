"""i25_fold.py -- train the I2.5 LEARNED FOLD (Deep-Sets set encoder) over the FROZEN RecVAE-d512.

Trains q(z | answer set) by reconstructing HELD-OUT liked interactions from SIMULATED partial reveals of
real TRAIN-user profiles: per user, shuffle rated items -> known / held-out halves; sample an interview-
length subset (k=1..16, dropout curriculum) of the known half as answer tokens (item real-rating +
concept/attribute data-side aggregates); the frozen RecVAE decoder scores; loss = multinomial
log-likelihood on the held-out likes (RecVAE's own objective). Answers are DATA-SIDE only -> the fold is
non-circular from birth. NO LLM calls. Saves ALL checkpoints to .cache/i25_fold_*; best on a DISJOINT val.

Run:  python scripts/i25_fold.py --n_users 25000 --epochs 12
"""
import os, sys, json, time, argparse, collections
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i25_lib as L

CKPT = ".cache/i25_fold.pt"
CKPT_BEST = ".cache/i25_fold_best.pt"
LOG = ".cache/i25_fold_log.json"


def make_user_split(prof, rng):
    """prof {j:r} -> (known {j:r}, heldout_likes set). Mirrors the eval half-split."""
    its = list(prof.keys()); rng.shuffle(its)
    h = len(its) // 2
    known = {j: prof[j] for j in its[:h]}
    held = set(j for j in its[h:] if prof[j] >= 4)
    return known, held


def build_reveal(D, FR, known, k, rng, channels):
    """Sample k known items -> revealed {j:r}; build tokens + native liked-item prior list."""
    ks = list(known.keys())
    if len(ks) > k:
        ks = [ks[i] for i in rng.choice(len(ks), size=k, replace=False)]
    revealed = {j: known[j] for j in ks}
    toks = L.build_tokens(D, FR, revealed, channels=channels)
    native = [j for j in ks if known[j] >= 4]
    return toks, native


def batch_loss(FR, model, D, users, rng, kmax=16, channels=("item", "concept", "attribute")):
    tok_lists, native_lists, targets, profs = [], [], [], []
    k = int(rng.integers(1, kmax + 1))                       # shared reveal size per step (curriculum)
    for u in users:
        known, held = u["known"], u["held"]
        if not held:
            continue
        toks, native = build_reveal(D, FR, known, k, rng, channels)
        if not toks:
            continue
        tok_lists.append(toks); native_lists.append(native)
        targets.append(held); profs.append(set(known.keys()))
    if not tok_lists:
        return None
    tt, tv, te, mask, nz = L.pack_batch(FR, tok_lists, native_lists)
    z = model(tt, tv, te, mask, nz)                          # (B,d)
    S = z @ FR.W.T + FR.bdec                                 # frozen decoder
    # mask profile (known) items out of the candidate set
    B, ni = S.shape
    negmask = torch.zeros((B, ni), dtype=torch.bool)
    tgt = torch.zeros((B, ni), dtype=torch.float32)
    for b in range(B):
        pj = list(profs[b])
        negmask[b, pj] = True
        tj = list(targets[b])
        tgt[b, tj] = 1.0
    S = S.masked_fill(negmask, -1e9)
    logp = torch.log_softmax(S, dim=1)
    n_t = tgt.sum(1).clamp(min=1)
    loss = -(logp * tgt).sum(1) / n_t                        # mean per-target multinomial LL
    return loss.mean()


def prep_users(prof_dict, rng):
    out = []
    for u, prof in prof_dict.items():
        known, held = make_user_split(prof, rng)
        if held and len(known) >= 2:
            out.append(dict(u=u, known=known, held=held))
    return out


@torch.no_grad()
def val_ndcg(FR, model, D, val_users, rng, channels=("item", "concept", "attribute")):
    """Fold the FULL known half of each val user; mean NDCG@10 full."""
    model.eval()
    vals = []
    for u in val_users:
        known, held = u["known"], u["held"]
        toks = L.build_tokens(D, FR, known, channels=channels)
        native = [j for j in known if known[j] >= 4]
        if not toks:
            continue
        z = L.fold_np(FR, model, toks, native)
        v = L.ndcg10(FR, z, held, set(known.keys()))
        if v is not None:
            vals.append(v)
    return float(np.mean(vals)) if vals else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_users", type=int, default=25000)
    ap.add_argument("--n_val", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--kmax", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    t0 = time.time()
    print(f"[i25] loading data + frozen RecVAE ...", flush=True)
    D = L.G.load_data()
    FR = L.Frozen(D)
    print(f"[i25] loading {args.n_users} train profiles ...", flush=True)
    prof = L.load_train_profiles(D, args.n_users + args.n_val, seed=args.seed)
    keys = list(prof.keys())
    rng = np.random.default_rng(args.seed); rng.shuffle(keys)
    val_keys = set(keys[:args.n_val]); tr_keys = keys[args.n_val:]
    split_rng = np.random.default_rng(args.seed + 1)
    tr_users = prep_users({u: prof[u] for u in tr_keys}, split_rng)
    val_users = prep_users({u: prof[u] for u in val_keys}, np.random.default_rng(args.seed + 2))
    print(f"[i25] train users {len(tr_users)}  val users {len(val_users)}  "
          f"(load {round(time.time()-t0,1)}s)", flush=True)

    model = L.Fold(D_LAT if (D_LAT := L.D_LAT) else 512)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    state = dict(epoch=0, best_val=-1.0, best_epoch=-1, history=[])
    if args.resume and os.path.exists(CKPT):
        blob = torch.load(CKPT, map_location="cpu")
        model.load_state_dict(blob["model"]); opt.load_state_dict(blob["opt"]); state = blob["state"]
        print(f"[i25] resumed at epoch {state['epoch']} best_val={state['best_val']:.4f}", flush=True)

    step_rng = np.random.default_rng(args.seed + 100)
    for ep in range(state["epoch"], args.epochs):
        model.train()
        order = np.arange(len(tr_users)); step_rng.shuffle(order)
        tot, nb = 0.0, 0
        for b0 in range(0, len(order), args.batch):
            us = [tr_users[i] for i in order[b0:b0 + args.batch]]
            loss = batch_loss(FR, model, D, us, step_rng, kmax=args.kmax)
            if loss is None:
                continue
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item()); nb += 1
        vN = val_ndcg(FR, model, D, val_users, step_rng)
        state["epoch"] = ep + 1
        rec = dict(epoch=ep + 1, train_loss=round(tot / max(nb, 1), 4), val_ndcg=round(vN, 4),
                   min=round((time.time() - t0) / 60, 1))
        state["history"].append(rec)
        print(f"[i25] ep {ep+1:2d} | loss {rec['train_loss']:.4f} | val NDCG@10 {vN:.4f} | "
              f"{rec['min']}m", flush=True)
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), state=state, args=vars(args)), CKPT)
        if vN > state["best_val"]:
            state["best_val"] = vN; state["best_epoch"] = ep + 1
            torch.save(dict(model=model.state_dict(), state=state, args=vars(args)), CKPT_BEST)
        json.dump(state["history"], open(LOG, "w"), indent=1)
    print(f"[i25] BEST val NDCG@10 {state['best_val']:.4f} @ep{state['best_epoch']}  "
          f"-> {CKPT_BEST}  (wall {round((time.time()-t0)/60,1)}m)", flush=True)


if __name__ == "__main__":
    main()
