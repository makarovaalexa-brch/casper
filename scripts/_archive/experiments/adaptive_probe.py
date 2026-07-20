"""ADAPTIVE-PROBE (author's test, 2026-07-14).

THE QUESTION (author, verbatim): "if you ask people their favourite genre, cluster them by it, and rank
probe questions by entropy within those groups, will you get the same order of questions?"

WHY IT IS DECISIVE: it needs NO policy, NO RL, NO optimisation. So it CANNOT be confounded by a weak
static arm (the flaw that undermines STATIC8). It measures the one thing that matters, directly:
    DOES THE ANSWER TO Q1 CHANGE WHICH Q2 IS BEST?
This is exactly Naghshvar & Javidi (JSTSP 2013) Cor. 8: adaptivity gain > 0 IFF the most informative
question depends on which hypothesis is true.

DESIGN
  Q1  = "what is your favourite genre?" -> cluster users by the ANSWER (argmax over the 18 genre tags,
        using the DISTILLED ANSWERER's own values). No folding needed for Q1 -- it only partitions.
  Q2  = each of the 800 bank items in turn, folded COLD into a0c (our strongest recommender,
        full-profile NDCG@10 = 0.4961), scored by NDCG@10 against the user's held-liked items.
  Then: does each cluster's BEST Q2 differ, and what does that buy?

HONESTY GUARDS
  * SELECTION SPLIT: the best Q2 is chosen on a DISJOINT half of users and evaluated on the other half --
    for BOTH arms symmetrically. Without this the adaptive arm wins by overfitting the selection, which is
    exactly the artifact that has burned this project before.
  * REFUSALS COST THE TURN: if the answerer refuses (know==0 or val<0), the fold yields nothing -> the user
    is scored on the COLD (popularity) ranking. No dodging.
  * NO INVENTED VALUE MAP: the star rating is recovered from the answerer's own crval + the user's
    known-mean, then passed through the SAME scale_rating() a0c was trained with.
  * HARD RULE #1: all val users, all 800 bank candidates, all 18 genres. No sampling, no top-N.
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from signed_latent import load_arena_base, scale_rating, SignedAE, ndcg10   # noqa
import arena_core as AC                                                      # noqa

A0C = "C:/dev/phd/casper/.cache/signed_latent/a0c_best.pt"
RS = "C:/dev/phd/casper/.cache/rich_signal"
TAG_Q = "C:/dev/phd/casper/.cache/instrument2/tag_questions.json"
TAG_MEMB = "C:/dev/phd/casper/.cache/instrument2/tag_membership.json"
ITEM_LISTS = "C:/dev/phd/casper/.cache/instrument2/item_lists.json"
META = "C:/dev/phd/casper/data/movielens/.cache/ml25m/meta.npz"
OFF_ITEM = 1628                      # bank layout: concepts[0:1128] + entities[1128:1628] + items[1628:2428]
GENRES = ['action','adventure','animation','comedy','crime','documentary','drama','family','fantasy',
          'horror','musical','mystery','noir','romance','sci-fi','thriller','war','western']


def log(m):
    print(m, flush=True)


def main():
    base = load_arena_base()
    ni = base["ni"]
    d = np.load(META)                       # the FULL ratings (rat_by_u holds only the 1k arena cohort)
    uu = d["uu"].astype(np.int64); ii = d["ii"].astype(np.int64); rr = d["rr"].astype(np.float64)
    o = np.argsort(uu, kind="stable"); uu, ii, rr = uu[o], ii[o], rr[o]
    bnd = np.searchsorted(uu, np.arange(uu[-1] + 2))

    bank = np.array([int(x) for x in json.load(open(ITEM_LISTS))["lists"]["top800"]["ids"]], np.int64)
    log(f"[probe] ni={ni} bank={len(bank)}")

    # ---- the answerer's val cohort (ALL of it) ----
    uids = np.load(f"{RS}/mm_val_uids.npy")
    K = np.load(f"{RS}/mm_val_know.npy"); V = np.load(f"{RS}/mm_val_val.npy")
    CR = np.load(f"{RS}/mm_val_crval.npy")
    log(f"[probe] val users={len(uids)}  tables={K.shape}")

    # ---- genre MEMBERSHIP (for the IMPLICIT favourite-genre signal: what do they WATCH most?) ----
    tq = json.load(open(TAG_Q))["tags"]
    memb = json.load(open(TAG_MEMB))["membership"]
    gmem = []
    for g in GENRES:
        tid = next(t["tagId"] for t in tq if str(t.get("tag")).lower() == g)
        gmem.append(set(int(j) for j in memb.get(str(tid), []) if 0 <= j < ni))
    log(f"[probe] genre tags: {len(gmem)}  (member counts: {[len(m) for m in gmem]})")

    # ---- per-user known/held (SAME uid-seeded split the answerer tables were built with) ----
    held, kmean, keep, favg = [], [], [], []
    for r, u in enumerate(uids):
        st_, en_ = bnd[int(u)], bnd[int(u) + 1]
        its, rat = ii[st_:en_], rr[st_:en_]
        if len(its) < 8:
            continue
        ru = np.random.default_rng(AC.SEED * 1_000_003 + int(u))
        perm = ru.permutation(len(its)); half = len(its) // 2
        kn = {int(its[k]): float(rat[k]) for k in perm[:half]}
        hl = set(int(its[k]) for k in perm[half:] if rat[k] >= 4)
        if len(kn) < 4 or not hl:
            continue
        # AUTHOR'S Q1, IMPLICIT: favourite genre = the genre they WATCH MOST in their known half.
        # Real signal from their own data -- not a 4-level sampled ordinal (which ties across genres).
        kset = set(kn)
        cnts = [len(kset & m) for m in gmem]
        favg.append(int(np.argmax(cnts)) if max(cnts) > 0 else -1)
        held.append(hl); kmean.append(float(np.mean(list(kn.values())))); keep.append(r)
    keep = np.array(keep, np.int64); kmean = np.array(kmean, np.float64)
    log(f"[probe] usable users={len(keep)} (dropped {len(uids)-len(keep)} with no held-liked)")
    K, V, CR = K[keep], V[keep], CR[keep]
    N = len(keep)

    # ---- a0c ----
    t = SignedAE(ni, use_mask=True)
    t.load_state_dict(torch.load(A0C, map_location="cpu")["model"]); t.eval()
    Wd = t.decoder.weight.detach(); bd = t.decoder.bias.detach()

    # ---- precompute the ranking for every (bank item, half-star level) + the COLD ranking ----
    # a0c's input has ONE nonzero => only 800 x 10 distinct inputs exist. (Real ML ratings ARE half-stars.)
    stars = np.arange(0.5, 5.01, 0.5)                              # 10 levels
    TOPN = 12
    tops = np.zeros((len(bank), len(stars), TOPN), np.int64)
    with torch.no_grad():
        xv = torch.zeros((1, ni))
        z0 = t.encode(xv); s0 = (z0 @ Wd.T + bd)[0]                # COLD = the popularity intercept
        cold_top = torch.topk(s0, TOPN).indices.numpy()
        for bi, item in enumerate(bank):
            xs = torch.zeros((len(stars), ni))
            for li, st in enumerate(stars):
                xs[li, int(item)] = float(scale_rating(st))
            z = t.encode(xs); sc = z @ Wd.T + bd
            tops[bi] = torch.topk(sc, TOPN, dim=-1).indices.numpy()
            if bi % 200 == 0:
                log(f"  [probe] precomputed {bi}/{len(bank)}")
    log("[probe] rankings precomputed (800 items x 10 levels + cold)")

    # ---- ordinal level -> star, DERIVED FROM THE DATA (not invented) ----
    # crval is NaN except for RATED bank items (the passthrough). The arena BINS stars into the 4 levels
    # (3 if star>=4.5, 2 if >=3.5, 1 if >=2.5, else 0). Invert that empirically from the rated cells.
    kmv = kmean[:, None]
    star_all = CR + kmv                                    # (N,800); NaN where not rated
    lvl_all = V[:, OFF_ITEM:].astype(np.int64)             # ordinal 0..3, -1 = refusal
    L2S = np.zeros(4)
    for L in range(4):
        m = np.isfinite(star_all) & (lvl_all == L)
        L2S[L] = float(np.nanmean(star_all[m])) if m.any() else 3.0
    log(f"[probe] ordinal->star map DERIVED FROM DATA: {dict(zip(['hated','meh','liked','loved'], np.round(L2S,3)))}")

    # ---- NDCG[u, q]: cold-start, ONE question asked ----
    disc = 1.0 / np.log2(np.arange(2, TOPN + 2))
    ND = np.zeros((N, len(bank)), np.float32)
    for u in range(N):
        hl = held[u]
        idcg = disc[:min(10, len(hl))].sum()
        cold_nd = sum(disc[r] for r, it in enumerate([x for x in cold_top][:10]) if it in hl) / idcg
        krow = K[u, OFF_ITEM:]; vrow = V[u, OFF_ITEM:]
        ans = (krow >= 1) & (vrow >= 0)                            # answered; else REFUSAL -> cold
        # RATED items: the real star (crval + user's known-mean). UNRATED but answered: the empirical
        # level->star map. Refused: no fold at all (the turn is burned).
        sr = star_all[u]
        st = np.where(np.isfinite(sr), sr, L2S[np.clip(vrow, 0, 3)])
        st = np.clip(np.rint(st * 2) / 2.0, 0.5, 5.0)
        lv = np.clip(np.rint(st * 2).astype(np.int64) - 1, 0, 9)
        for bi in range(len(bank)):
            if not ans[bi]:
                ND[u, bi] = cold_nd; continue
            rank = [x for x in tops[bi, lv[bi]] if x != int(bank[bi])][:10]   # asked item is now known
            ND[u, bi] = sum(disc[r] for r, it in enumerate(rank) if it in hl) / idcg
        if u % 500 == 0:
            log(f"  [probe] scored {u}/{N} users")

    np.save(".cache/adaptive_probe_ND.npy", ND)
    np.save(".cache/adaptive_probe_fav.npy", np.array(favg, np.int64))
    log("[probe] cached ND + fav")

    # ---- Q1: favourite genre = IMPLICIT, what they WATCH MOST (author's correction) ----
    # NOT the answerer's 4-level ordinal (which ties across genres, making argmax arbitrary noise).
    fav = np.array(favg, np.int64)
    log("\n[probe] CLUSTERS (favourite genre = most-watched in the known half):")
    for c, n in zip(*np.unique(fav, return_counts=True)):
        log(f"    {('no-genre' if c < 0 else GENRES[c]):<12} n={n}")

    # ---- SELECTION SPLIT (the honesty guard): pick the best Q2 on half the users, EVALUATE on the other ----
    rng = np.random.default_rng(0); perm = rng.permutation(N)
    SEL, EVA = perm[:N // 2], perm[N // 2:]

    gq = int(np.argmax(ND[SEL].mean(0)))                           # ONE question for everybody (static)
    static_score = float(ND[EVA, gq].mean())

    rows, adaptive_num, adaptive_den = [], 0.0, 0
    for c in np.unique(fav):
        sel_c = SEL[fav[SEL] == c]; eva_c = EVA[fav[EVA] == c]
        if len(sel_c) < 20 or len(eva_c) < 20:
            continue
        bq = int(np.argmax(ND[sel_c].mean(0)))                     # this cluster's OWN best question
        sc = float(ND[eva_c, bq].mean())
        adaptive_num += sc * len(eva_c); adaptive_den += len(eva_c)
        name = "no-answer" if c < 0 else GENRES[c]
        rows.append((name, len(eva_c), bq, int(bank[bq]), sc, float(ND[eva_c, gq].mean())))
    adaptive_score = adaptive_num / max(adaptive_den, 1)

    log("\n" + "=" * 100)
    log(f"{'cluster':<12} {'n_eval':>7} {'its best Q2':>12} {'NDCG(own Q2)':>13} {'NDCG(global Q2)':>16} {'gain':>8}")
    log("-" * 100)
    for name, n, bq, item, sc, gsc in sorted(rows, key=lambda r: -r[1]):
        log(f"{name:<12} {n:>7} {bq:>12} {sc:>13.4f} {gsc:>16.4f} {sc-gsc:>+8.4f}")
    log("=" * 100)
    nd = len({r[2] for r in rows})
    log(f"\nDISTINCT best-Q2 across {len(rows)} clusters: {nd}")
    log(f"  (if 1 => every cluster wants the SAME next question => ADAPTIVITY IS WORTH ZERO, by direct measurement)")
    log(f"\nSTATIC   (everyone asked the single globally-best Q2): {static_score:.4f}")
    log(f"ADAPTIVE (each cluster asked ITS OWN best Q2)        : {adaptive_score:.4f}")
    log(f"** ADAPTIVE PRIZE AT TURN 2 = {adaptive_score - static_score:+.4f} **")
    log("  (selection on a DISJOINT half of users, evaluated on the other half, symmetric for both arms)")

    # ---- ranking agreement: do the clusters even ORDER the questions the same way? ----
    from itertools import combinations
    ords = {}
    for c in np.unique(fav):
        sel_c = SEL[fav[SEL] == c]
        if len(sel_c) < 20:
            continue
        ords[c] = ND[sel_c].mean(0)
    ks = list(ords)
    taus, ov = [], []
    for a, b in combinations(ks, 2):
        ra = np.argsort(-ords[a]); rb = np.argsort(-ords[b])
        pa = np.empty(len(bank)); pa[ra] = np.arange(len(bank))
        pb = np.empty(len(bank)); pb[rb] = np.arange(len(bank))
        taus.append(np.corrcoef(pa, pb)[0, 1])                     # Spearman on ranks
        ov.append(len(set(ra[:10]) & set(rb[:10])) / 10.0)
    log(f"\nRANK AGREEMENT BETWEEN CLUSTERS (do they want the same question ORDER?)")
    log(f"  Spearman(rank_a, rank_b)  mean={np.mean(taus):.4f}  min={np.min(taus):.4f}  max={np.max(taus):.4f}")
    log(f"  top-10 overlap            mean={np.mean(ov):.3f}   min={np.min(ov):.3f}")
    log(f"  (Spearman ~1.0 and overlap ~1.0 => THE SAME ORDER FOR EVERYBODY => the author's question answered NO)")

    # ---- SIGNIFICANCE: paired bootstrap over EVAL users (the arms are paired per user) ----
    per_u_ad = np.zeros(len(EVA)); per_u_st = ND[EVA, gq]
    bestq = {}
    for c in np.unique(fav):
        sel_c = SEL[fav[SEL] == c]
        if len(sel_c) >= 20:
            bestq[c] = int(np.argmax(ND[sel_c].mean(0)))
    ok = np.array([fav[u] in bestq for u in EVA])
    for i, u in enumerate(EVA):
        per_u_ad[i] = ND[u, bestq[fav[u]]] if fav[u] in bestq else np.nan
    d = (per_u_ad - per_u_st)[ok]
    rng2 = np.random.default_rng(0)
    bs = np.array([d[rng2.integers(0, len(d), len(d))].mean() for _ in range(5000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    log("")
    log(f"PAIRED BOOTSTRAP (n={len(d)} eval users, 5000 resamples)")
    log(f"  adaptive - static = {d.mean():+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   "
        f"{'SIGNIFICANT' if lo > 0 else 'NOT significant (CI spans 0)'}")
    log(f"  P(adaptive > static) per user = {(d > 0).mean():.3f}   P(equal) = {(d == 0).mean():.3f}")


if __name__ == "__main__":
    main()
