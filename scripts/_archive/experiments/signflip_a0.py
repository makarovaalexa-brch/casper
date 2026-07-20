"""QUICK sign-flip diagnostic on a0c (alpha=0 signed-latent model). No training.
Does the model USE the sign of the input? Flip revealed answers +->- for ITEMS and GENRES,
measure whether the output ranking of the flipped entity's region moves DOWN.
"""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
import numpy as np
import torch
torch.set_num_threads(2)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from signed_latent import (load_arena_base, SignedAE, build_gmat, OUTDIR)

def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

CKPT = "C:/dev/phd/casper/.cache/signed_latent/a0c_best.pt"


def score(model, ni, idx, val):
    xv = np.zeros(ni, np.float32)
    xv[np.asarray(idx, np.int64)] = np.asarray(val, np.float32)
    with torch.no_grad():
        return model(torch.from_numpy(xv[None, :]), 0.0)[0].numpy()[0].astype(np.float64)


def pct_ranks(scores):
    """percentile (0..1, 1=top score) of every item."""
    order = np.argsort(scores)
    r = np.empty(len(scores)); r[order] = np.arange(len(scores)) / (len(scores) - 1)
    return r


def mean_pct(scores, items, exclude):
    pr = pct_ranks(scores)
    items = [i for i in items if i not in exclude]
    return float(np.mean(pr[np.asarray(items, np.int64)])), items


def topk_overlap(a, b, exclude, k=10):
    a = a.copy(); b = b.copy()
    ex = list(exclude)
    a[ex] = -1e30; b[ex] = -1e30
    ta = set(np.argsort(-a)[:k].tolist()); tb = set(np.argsort(-b)[:k].tolist())
    return len(ta & tb) / k


def main():
    base = load_arena_base(); ni = base["ni"]; cnt = base["cnt"]
    model = SignedAE(ni, use_mask=True)
    blob = torch.load(CKPT, map_location="cpu")
    model.load_state_dict(blob["model"]); model.eval()
    log(f"loaded a0c val_full={blob.get('val_full')} alpha={blob.get('alpha')}")

    Gmat, gix = build_gmat(base)
    d = np.load("C:/dev/phd/casper/data/movielens/.cache/ml25m/meta.npz")
    keepI = d["keepI"].astype(np.int64)
    # title lookup
    title = {}
    with open("data/movielens/movies.csv", encoding="utf-8") as f:
        next(f)
        for line in f:
            i0 = line.find(","); i1 = line.rfind(",")
            try: mid = int(line[:i0])
            except ValueError: continue
            title[mid] = line[i0+1:i1].strip('"')
    def tt(i): return title.get(int(keepI[i]), f"item{i}")

    out = {"ckpt": CKPT, "val_full": blob.get("val_full"), "ni": ni}

    # ---------------- ITEMS ----------------
    log("=== ITEM sign-flip ===")
    pop_order = np.argsort(-cnt)
    probe_items = pop_order[:10].tolist()   # 10 most popular items
    # nearest neighbors via decoder weight cosine (item region)
    W = model.decoder.weight.detach().numpy()  # (ni, latent)
    Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
    item_res = []
    for it in probe_items:
        # nearest neighbors of the item in decoder-embedding space (its region)
        sims = Wn @ Wn[it]
        nn = np.argsort(-sims)[1:21]  # top-20 neighbors (excl self)
        s_like = score(model, ni, [it], [1.0])
        s_dis = score(model, ni, [it], [-1.0])
        exclude = {it}
        nn_like, _ = mean_pct(s_like, nn.tolist(), exclude)
        nn_dis, _ = mean_pct(s_dis, nn.tolist(), exclude)
        ov = topk_overlap(s_like, s_dis, exclude, 10)
        # rank correlation (spearman via percentile arrays)
        rl = pct_ranks(s_like); rd = pct_ranks(s_dis)
        rho = float(np.corrcoef(rl, rd)[0, 1])
        item_res.append(dict(item=int(it), title=tt(it), pop=int(cnt[it]),
                             nn_pct_like=nn_like, nn_pct_dislike=nn_dis,
                             nn_drop=nn_like - nn_dis, top10_overlap=ov, rank_corr=rho))
        log(f"  {tt(it)[:34]:34s} nn_like={nn_like:.3f} nn_dis={nn_dis:.3f} "
            f"drop={nn_like-nn_dis:+.3f} ov10={ov:.2f} rho={rho:.3f}")
    out["items"] = item_res
    out["item_mean_nn_drop"] = float(np.mean([r["nn_drop"] for r in item_res]))
    out["item_mean_top10_overlap"] = float(np.mean([r["top10_overlap"] for r in item_res]))
    out["item_mean_rank_corr"] = float(np.mean([r["rank_corr"] for r in item_res]))

    # ---------------- GENRES (member-bag) ----------------
    log("=== GENRE (member-bag) sign-flip ===")
    probe_genres = ["Sci-Fi", "Horror", "Romance", "Documentary", "Children", "War", "Comedy", "Action"]
    NP = 20   # member items revealed as the answer bag (popular members)
    genre_res = {}
    for g in probe_genres:
        gi = gix[g]; members = np.where(Gmat[:, gi] > 0)[0]
        pop_mem = members[np.argsort(-cnt[members])]
        bag = pop_mem[:NP]                       # revealed (input)
        held = pop_mem[NP:NP + 300]              # measured (not in input)
        s_like = score(model, ni, bag, np.ones(NP))
        s_dis = score(model, ni, bag, -np.ones(NP))
        exclude = set(bag.tolist())
        like_pct, _ = mean_pct(s_like, held.tolist(), exclude)
        dis_pct, _ = mean_pct(s_dis, held.tolist(), exclude)
        # specificity: untouched genres' members displacement
        ctrl_disp = []
        for cg in probe_genres:
            if cg == g: continue
            cmem = np.where(Gmat[:, gix[cg]] > 0)[0]
            cmem = cmem[np.argsort(-cnt[cmem])][:300]
            cl, _ = mean_pct(s_like, cmem.tolist(), exclude)
            cd, _ = mean_pct(s_dis, cmem.tolist(), exclude)
            ctrl_disp.append(abs(cl - cd))
        genre_res[g] = dict(like_pct=like_pct, dislike_pct=dis_pct, drop=like_pct - dis_pct,
                            mean_ctrl_displacement=float(np.mean(ctrl_disp)))
        log(f"  {g:12s} like={like_pct:.3f} dislike={dis_pct:.3f} DROP={like_pct-dis_pct:+.3f} "
            f"ctrl_disp={np.mean(ctrl_disp):.3f}")
    out["genres"] = genre_res
    out["genre_mean_drop"] = float(np.mean([v["drop"] for v in genre_res.values()]))
    out["genre_mean_ctrl_displacement"] = float(np.mean([v["mean_ctrl_displacement"] for v in genre_res.values()]))

    # ---------------- VERDICT ----------------
    gdrop = out["genre_mean_drop"]; idrop = out["item_mean_nn_drop"]
    iov = out["item_mean_top10_overlap"]
    resp_g = gdrop > 0.03
    resp_i = (idrop > 0.03) or (iov < 0.8)
    out["verdict"] = {
        "genre_responds_to_sign": bool(resp_g),
        "item_responds_to_sign": bool(resp_i),
        "genre_mean_drop": gdrop,
        "item_mean_nn_drop": idrop,
        "item_mean_top10_overlap": iov,
    }
    json.dump(out, open(os.path.join(OUTDIR, "signflip_a0.json"), "w"), indent=2)
    log(f"WROTE {os.path.join(OUTDIR, 'signflip_a0.json')}")
    log(f"VERDICT genre_drop={gdrop:+.3f} (responds={resp_g}) | item_nn_drop={idrop:+.3f} ov10={iov:.2f} (responds={resp_i})")


if __name__ == "__main__":
    main()
