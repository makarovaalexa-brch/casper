"""ADAPTIVE-PROBE v2 — the author's test, done PROPERLY (2026-07-14).

v1 flaws the author caught, all fixed here:
  1. TINY CLUSTERS: v1 used only the 3,000-user mm_val cohort => family n=44, adventure self-agreement 0.00.
     We have 150,000 MORE labelled users in mm_train. v2 uses ALL 153,000.
  2. NO TAIL: v1 reported full NDCG@10 only. TAIL is the headline metric of Papers B/C (and in STATIC8 the
     adaptive gap was BIGGER on tail). v2 reports BOTH, using the arena's exact tail definition
     (head items masked out of the ranking; relevance = held-liked NON-head items).
  3. WHICH QUESTIONS? v1 never said. v2 prints the actual movie titles.
  4. ONE TURN: this is still a 1-question probe (Q1 partitions, Q2 is scored). STATED, not hidden.
     Flatness at turn 1 (cold start, any popular film tells you something) may be nothing like flatness at
     turn 6, when the belief is sharp. That is the next experiment, not this one.

DESIGN. Q1 = favourite genre, IMPLICIT (the genre the user WATCHES MOST in their known half).
Q2 = each of the 800 bank items, folded COLD into a0c (full-profile NDCG@10 = 0.4961), scored vs held-liked.
GUARD: the best Q2 is chosen on a DISJOINT half of users and evaluated on the other half, symmetrically for
both arms. HARD RULE #1: all 153k users, all 800 candidates, all 18 genres. No sampling.
"""
import os, sys, json, csv
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from signed_latent import load_arena_base, scale_rating, SignedAE   # noqa
import arena_core as AC                                             # noqa

A0C = "C:/dev/phd/casper/.cache/signed_latent/a0c_best.pt"
RS = "C:/dev/phd/casper/.cache/rich_signal"
TAG_Q = "C:/dev/phd/casper/.cache/instrument2/tag_questions.json"
TAG_MEMB = "C:/dev/phd/casper/.cache/instrument2/tag_membership.json"
ITEM_LISTS = "C:/dev/phd/casper/.cache/instrument2/item_lists.json"
META = "C:/dev/phd/casper/data/movielens/.cache/ml25m/meta.npz"
MOVIES = "C:/dev/phd/casper/data/movielens/movies.csv"
OFF_ITEM = 1628
GENRES = ['action','adventure','animation','comedy','crime','documentary','drama','family','fantasy',
          'horror','musical','mystery','noir','romance','sci-fi','thriller','war','western']
TOPN = 14


def log(m):
    print(m, flush=True)


def main():
    base = load_arena_base()
    ni = base["ni"]; head = base["headmask"]
    d = np.load(META)
    uu = d["uu"].astype(np.int64); ii = d["ii"].astype(np.int64); rr = d["rr"].astype(np.float64)
    o = np.argsort(uu, kind="stable"); uu, ii, rr = uu[o], ii[o], rr[o]
    bnd = np.searchsorted(uu, np.arange(uu[-1] + 2))
    keepI = d["keepI"].astype(np.int64) if "keepI" in d else None

    bank = np.array([int(x) for x in json.load(open(ITEM_LISTS))["lists"]["top800"]["ids"]], np.int64)
    nb = len(bank)

    # titles
    title = {}
    if keepI is not None and os.path.exists(MOVIES):
        mid2t = {}
        with open(MOVIES, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                mid2t[int(row["movieId"])] = row["title"]
        for dense, mid in enumerate(keepI):
            if dense < ni:
                title[dense] = mid2t.get(int(mid), f"item{dense}")
    tt = lambda j: title.get(int(j), f"item{j}")

    # ---- ALL labelled users: 150k train + 3k val (the author's point: we labelled all of it) ----
    U, K, V, CR = [], [], [], []
    for tag in ("train", "val"):
        U.append(np.load(f"{RS}/mm_{tag}_uids.npy"))
        K.append(np.load(f"{RS}/mm_{tag}_know.npy")); V.append(np.load(f"{RS}/mm_{tag}_val.npy"))
        CR.append(np.load(f"{RS}/mm_{tag}_crval.npy"))
    uids = np.concatenate(U); K = np.concatenate(K); V = np.concatenate(V); CR = np.concatenate(CR)
    log(f"[probe2] labelled users = {len(uids)}  (train {len(U[0])} + val {len(U[1])})   bank = {nb}")

    tq = json.load(open(TAG_Q))["tags"]; memb = json.load(open(TAG_MEMB))["membership"]
    gmem = []
    for g in GENRES:
        tid = next(t["tagId"] for t in tq if str(t.get("tag")).lower() == g)
        gmem.append(np.array(sorted(int(j) for j in memb.get(str(tid), []) if 0 <= j < ni), np.int64))
    gmask = np.zeros((len(GENRES), ni), bool)
    for gi, m in enumerate(gmem):
        gmask[gi, m] = True

    # ---- per-user: held-liked (full + tail), known-mean, implicit favourite genre ----
    held_f, held_t, kmean, keep, favg = [], [], [], [], []
    for r, u in enumerate(uids):
        a, b = bnd[int(u)], bnd[int(u) + 1]
        its, rat = ii[a:b], rr[a:b]
        if len(its) < 8:
            continue
        ru = np.random.default_rng(AC.SEED * 1_000_003 + int(u))
        perm = ru.permutation(len(its)); h = len(its) // 2
        ki, kr = its[perm[:h]], rat[perm[:h]]
        hi, hr = its[perm[h:]], rat[perm[h:]]
        hl = hi[hr >= 4]
        if len(ki) < 4 or len(hl) == 0:
            continue
        hlt = hl[~head[hl]]
        if len(hlt) == 0:
            continue                                     # tail metric undefined for this user
        cnts = gmask[:, ki].sum(1)
        favg.append(int(np.argmax(cnts)) if cnts.max() > 0 else -1)
        held_f.append(hl); held_t.append(hlt); kmean.append(float(kr.mean())); keep.append(r)
        if len(keep) % 25000 == 0:
            log(f"  [probe2] prepared {len(keep)} users")
    keep = np.array(keep, np.int64); kmean = np.array(kmean)
    K, V, CR = K[keep], V[keep], CR[keep]
    fav = np.array(favg, np.int64); N = len(keep)
    log(f"[probe2] usable users = {N}")
    for c, n in zip(*np.unique(fav, return_counts=True)):
        log(f"    {('no-genre' if c < 0 else GENRES[c]):<12} n={n}")

    # ---- a0c: precompute the FULL and TAIL rankings for every (bank item, half-star level) ----
    t = SignedAE(ni, use_mask=True)
    t.load_state_dict(torch.load(A0C, map_location="cpu")["model"]); t.eval()
    Wd = t.decoder.weight.detach(); bd = t.decoder.bias.detach()
    hmask = torch.from_numpy(head)
    stars = np.arange(0.5, 5.01, 0.5)
    topF = np.zeros((nb, 10, TOPN), np.int64); topT = np.zeros((nb, 10, TOPN), np.int64)
    with torch.no_grad():
        z0 = t.encode(torch.zeros((1, ni))); s0 = (z0 @ Wd.T + bd)[0]
        coldF = torch.topk(s0, TOPN).indices.numpy()
        s0t = s0.clone(); s0t[hmask] = -1e30
        coldT = torch.topk(s0t, TOPN).indices.numpy()
        for bi, item in enumerate(bank):
            xs = torch.zeros((10, ni))
            for li, st in enumerate(stars):
                xs[li, int(item)] = float(scale_rating(st))
            sc = t.encode(xs) @ Wd.T + bd
            topF[bi] = torch.topk(sc, TOPN, dim=-1).indices.numpy()
            sct = sc.clone(); sct[:, hmask] = -1e30
            topT[bi] = torch.topk(sct, TOPN, dim=-1).indices.numpy()
            if bi % 200 == 0:
                log(f"  [probe2] precomputed {bi}/{nb}")

    # ---- ordinal -> star, DERIVED FROM THE DATA ----
    star_all = CR + kmean[:, None]
    lvl_all = V[:, OFF_ITEM:].astype(np.int64)
    L2S = np.zeros(4)
    for L in range(4):
        m = np.isfinite(star_all) & (lvl_all == L)
        L2S[L] = float(np.nanmean(star_all[m])) if m.any() else 3.0
    log(f"[probe2] ordinal->star (from data): {dict(zip(['hated','meh','liked','loved'], np.round(L2S, 3)))}")

    W = 1.0 / np.log2(np.arange(2, TOPN + 2))
    rng = np.random.default_rng(0); half = rng.random(N) < 0.5      # SELECTION half vs EVAL half

    C = len(GENRES)
    accS = np.zeros((C + 1, nb)); accE = np.zeros((C + 1, nb))
    cntS = np.zeros(C + 1); cntE = np.zeros(C + 1)
    accS_t = np.zeros((C + 1, nb)); accE_t = np.zeros((C + 1, nb))
    oraF = np.zeros(N); oraT = np.zeros(N)
    ndF_u = np.zeros((N, nb), np.float16); ndT_u = np.zeros((N, nb), np.float16)

    kr = K[:, OFF_ITEM:]; vr = V[:, OFF_ITEM:]
    for u in range(N):
        relF = np.zeros(ni, bool); relF[held_f[u]] = True
        relT = np.zeros(ni, bool); relT[held_t[u]] = True
        idF = W[:min(10, len(held_f[u]))].sum(); idT = W[:min(10, len(held_t[u]))].sum()
        sr = star_all[u]
        st = np.where(np.isfinite(sr), sr, L2S[np.clip(vr[u], 0, 3)])
        lv = np.clip(np.rint(np.clip(np.rint(st * 2) / 2.0, 0.5, 5.0) * 2).astype(np.int64) - 1, 0, 9)
        ans = (kr[u] >= 1) & (vr[u] >= 0)

        tf = topF[np.arange(nb), lv]; tt_ = topT[np.arange(nb), lv]           # (nb, TOPN)
        tf = np.where(ans[:, None], tf, coldF[None, :])                       # refusal -> cold
        tt_ = np.where(ans[:, None], tt_, coldT[None, :])
        okf = tf != bank[:, None]; okt = tt_ != bank[:, None]                 # drop the asked item
        def dcg(top, ok, rel, idcg):
            r = rel[top] & ok
            pos = np.cumsum(ok, 1) - 1
            m = ok & (pos < 10)
            return (np.where(m & r, W[np.clip(pos, 0, TOPN - 1)], 0.0).sum(1)) / idcg
        nf = dcg(tf, okf, relF, idF); nt = dcg(tt_, okt, relT, idT)
        ndF_u[u] = nf.astype(np.float16); ndT_u[u] = nt.astype(np.float16)
        oraF[u] = nf.max(); oraT[u] = nt.max()
        c = fav[u] if fav[u] >= 0 else C
        if half[u]:
            accS[c] += nf; accS_t[c] += nt; cntS[c] += 1
        else:
            accE[c] += nf; accE_t[c] += nt; cntE[c] += 1
        if u % 20000 == 0:
            log(f"  [probe2] scored {u}/{N}")

    np.save(".cache/probe2_ndF.npy", ndF_u); np.save(".cache/probe2_ndT.npy", ndT_u)
    np.save(".cache/probe2_fav.npy", fav); np.save(".cache/probe2_half.npy", half)

    for NAME, aS, aE, ora in (("FULL", accS, accE, oraF), ("TAIL", accS_t, accE_t, oraT)):
        gS = aS.sum(0) / max(cntS.sum(), 1)
        gq = int(np.argmax(gS))
        gE = aE.sum(0) / max(cntE.sum(), 1)
        log("\n" + "=" * 108)
        log(f"  {NAME} NDCG@10   |   GLOBAL best Q2 = [{gq}] {tt(bank[gq])}")
        log("=" * 108)
        log(f"{'cluster':<12} {'n_eval':>8} {'its best Q2 (chosen on the OTHER half)':<46} {'own':>7} {'global':>8} {'gain':>8}")
        log("-" * 108)
        num = den = 0.0
        for c in range(C + 1):
            if cntS[c] < 50 or cntE[c] < 50:
                continue
            bq = int(np.argmax(aS[c] / cntS[c]))
            own = aE[c][bq] / cntE[c]; glo = aE[c][gq] / cntE[c]
            num += own * cntE[c]; den += cntE[c]
            nm = "no-genre" if c == C else GENRES[c]
            log(f"{nm:<12} {int(cntE[c]):>8} {('[' + str(bq) + '] ' + tt(bank[bq]))[:45]:<46} "
                f"{own:>7.4f} {glo:>8.4f} {own-glo:>+8.4f}")
        adaptive = num / max(den, 1); static = float(gE[gq])
        log("-" * 108)
        log(f"STATIC  (everyone -> the single globally-best Q2): {static:.4f}")
        log(f"ADAPTIVE (each cluster -> its own best Q2)       : {adaptive:.4f}")
        log(f"** ADAPTIVE PRIZE ({NAME}) = {adaptive - static:+.4f} **")
        log(f"CLAIRVOYANT CEILING (each user -> THEIR own best Q2, knows the outcome): {ora.mean():.4f} "
            f"(+{ora.mean()-static:.4f})")
        log(f"\n  flatness of the {NAME} objective (population mean by question rank):")
        gsort = np.argsort(-gS)
        log("   " + "  ".join(f"r{r+1}:{gS[gsort[r]]:.4f}" for r in [0, 1, 4, 9, 19, 49, 99]))
        log(f"\n  TOP 5 QUESTIONS ({NAME}, population):")
        for r in range(5):
            q = gsort[r]
            log(f"    {r+1}. [{q}] {tt(bank[q])}   NDCG {gS[q]:.4f}")


if __name__ == "__main__":
    main()
