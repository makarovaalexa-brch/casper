"""recvae_valence_probe.py -- CHEAP, NO-TRAINING probe of the FROZEN RecVAE-d512 latent geometry.

Question: does the frozen RecVAE z-space contain a VALENCE direction -- can "dislike X" geometrically
DOWN-RANK region X BELOW neutral, or does it only encode consumption (so dislike cannot suppress)?

Reuses arena/gate convention verbatim:
  - cold/population belief   z0 = zeros  (decode(z0) = decoder bias bdec = popularity prior)
  - belief operator          z' = z0 + eta*a*d_X ,  a=+1 like / a=-1 dislike
  - eta                      canonical ETA=16 (llm_answerability_gate.ETA / ml25m_gates G8), plus sweep
  - d_X (concept direction)  unit member-bag encoding of the genre members (Frozen.attr_emb convention)
  - decode                   Frozen.decode_np(z) = z @ W.T + bdec

NO training, NO LLM, NO data reduction. Loads the certified frozen checkpoint only.
"""
import os, sys, json
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "instrument2"))
import warnings
warnings.filterwarnings("ignore")

import llm_answerability_gate as G
import i25_lib as L

ETA_MAIN = float(G.ETA)                      # 16.0 canonical belief-operator scale
ETA_SWEEP = [4.0, 8.0, 16.0, 32.0]
OUT = ".cache/arena/recvae_valence_probe.json"
GENRE_NAMES = ["Sci-Fi", "Horror", "Romance", "Comedy", "Animation", "Documentary", "War", "Musical"]


def pct_of(scores_all, member_ids):
    """Mean score-PERCENTILE (0..100) of member items among ALL catalogue items."""
    ss = np.sort(scores_all)
    N = len(ss)
    ranks = np.searchsorted(ss, scores_all[member_ids], side="left")
    return float((ranks / (N - 1) * 100.0).mean())


def genre_members(D, gname):
    gi = G.GIX[gname]
    return gi, np.where(D["Gmat"][:, gi] > 0)[0]


def title_line(D, j):
    gens = [G.GENRES[k] for k in np.where(D["Gmat"][j] > 0)[0]]
    return f"{D['title'][j]}  [{', '.join(gens)}]"


def find_item(D, needle):
    nl = needle.lower()
    hits = [j for j in range(int(D["ni"])) if nl in D["title"][j].lower()]
    # prefer the most popular match (well-known)
    hits.sort(key=lambda j: -D["cnt"][j])
    return hits[0] if hits else None


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    print("[probe] loading data + frozen RecVAE-d512 ...", flush=True)
    D = G.load_data()
    FR = L.Frozen(D)
    ni = int(D["ni"])
    bdec = FR.bdec.numpy().astype(np.float64)
    W = FR.W.numpy().astype(np.float64)                       # (ni, d)

    z0 = np.zeros(W.shape[1], np.float64)                     # cold / population prior
    s_neutral_all = FR.decode_np(z0[None, :])[0]              # == bdec

    genres = {}
    for gname in GENRE_NAMES:
        gi, mem = genre_members(D, gname)
        d_X = FR.attr_emb(("gen", gi)).astype(np.float64)     # unit member-bag concept direction
        neu = pct_of(s_neutral_all, mem)
        sweep = {}
        for eta in ETA_SWEEP:
            s_like = FR.decode_np((z0 + eta * d_X)[None, :])[0]
            s_dis = FR.decode_np((z0 - eta * d_X)[None, :])[0]
            sweep[f"{eta:g}"] = dict(like_pct=pct_of(s_like, mem),
                                     dislike_pct=pct_of(s_dis, mem))
        # headline eta
        s_like = FR.decode_np((z0 + ETA_MAIN * d_X)[None, :])[0]
        s_dis = FR.decode_np((z0 - ETA_MAIN * d_X)[None, :])[0]
        like = pct_of(s_like, mem); dis = pct_of(s_dis, mem)
        genres[gname] = dict(n_members=int(len(mem)), neutral_pct=neu,
                             like_pct=like, dislike_pct=dis,
                             elevate_gap=like - neu, suppress_gap=neu - dis,
                             sweep=sweep)
        print(f"  {gname:12s} n={len(mem):5d}  like {like:6.2f}  neutral {neu:6.2f}  "
              f"dislike {dis:6.2f}  | +{like-neu:6.2f} / -{neu-dis:6.2f}", flush=True)

    # ---- IG1 genre purity: top-10 under LIKE for 2 genres (Sci-Fi, Animation) ----
    purity = {}
    for gname in ("Sci-Fi", "Animation"):
        gi, _ = genre_members(D, gname)
        d_X = FR.attr_emb(("gen", gi)).astype(np.float64)
        s = FR.decode_np((z0 + ETA_MAIN * d_X)[None, :])[0]
        top = np.argsort(-s)[:10]
        purity[gname] = [title_line(D, int(j)) for j in top]

    # ---- IG3 franchise/item coherence: 3 films, z toward that single item (native encode) ----
    franchise = {}
    for needle in ("Star Wars", "Inception", "Toy Story"):
        j = find_item(D, needle)
        if j is None:
            franchise[needle] = ["<not found>"]; continue
        z = FR.native_fold_np([int(j)])                       # RecVAE native fold of the single item
        s = FR.decode_np(z[None, :])[0]
        s[j] = -1e9                                            # exclude the seed itself
        top = np.argsort(-s)[:10]
        franchise[f"{needle} -> {D['title'][j]}"] = [title_line(D, int(k)) for k in top]

    out = dict(eta_main=ETA_MAIN, eta_sweep=ETA_SWEEP, z0="zeros (cold/population prior)",
               convention="z'=z0+eta*a*d_X ; d_X=unit member-bag genre dir ; decode=z@W.T+bdec",
               genres=genres, purity=purity, franchise=franchise)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\n[probe] wrote {OUT}", flush=True)

    # ---- print deliverables ----
    print("\n=== (a) PER-GENRE VALENCE TABLE (percentile of member items; eta=%.0f) ===" % ETA_MAIN)
    print(f"{'genre':12s} {'like':>7s} {'neutral':>8s} {'dislike':>8s} {'elevate':>8s} {'suppress':>9s}")
    for g, r in genres.items():
        print(f"{g:12s} {r['like_pct']:7.2f} {r['neutral_pct']:8.2f} {r['dislike_pct']:8.2f} "
              f"{r['elevate_gap']:+8.2f} {r['suppress_gap']:+9.2f}")

    print("\n=== (b) GENRE PURITY: top-10 under LIKE ===")
    for g, lst in purity.items():
        print(f"-- LIKE {g} -->")
        for t in lst:
            print(f"     {t}")

    print("\n=== (c) FRANCHISE COHERENCE: top-10 nearest to a single item ===")
    for k, lst in franchise.items():
        print(f"-- {k} -->")
        for t in lst:
            print(f"     {t}")

    # ---- verdict ----
    elev = np.mean([r["elevate_gap"] for r in genres.values()])
    supp = np.mean([r["suppress_gap"] for r in genres.values()])
    n_real_supp = sum(1 for r in genres.values() if r["suppress_gap"] >= 5.0)
    print("\n=== (d) SYMMETRY + VERDICT ===")
    print(f"mean elevate gap (like-neutral)  = {elev:+.2f} pct-pts")
    print(f"mean suppress gap (neutral-dislike) = {supp:+.2f} pct-pts")
    print(f"genres with real suppression (>=5 pct-pts below neutral): {n_real_supp}/{len(genres)}")
    if supp >= 5.0 and n_real_supp >= len(genres) * 0.6:
        verdict = "YES"
    elif supp >= 2.0 and n_real_supp >= 2:
        verdict = "WEAK"
    else:
        verdict = "NO"
    print(f"\nVERDICT: RecVAE frozen z-space usable valence direction (dislike actively suppresses)? "
          f"{verdict}  (elevate {elev:+.1f} vs suppress {supp:+.1f} pct-pts; "
          f"{'symmetric' if abs(elev-supp) < 0.4*abs(elev) else 'suppression << elevation' if supp < elev else 'asym'})")
    out["verdict"] = dict(verdict=verdict, mean_elevate=float(elev), mean_suppress=float(supp),
                          n_real_suppress=int(n_real_supp), n_genres=len(genres))
    json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
