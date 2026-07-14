"""ENTROPY vs NDCG — the boundary demonstration (author's question, 2026-07-14).

THE QUESTION (author): *"is it worth looking at entropy, not NDCG optimisation directly?"*

THE THEORY SAYS THEY DIVERGE, AND THAT THIS IS THE WHOLE STORY:
  - Krause & Guestrin (ICML 2007): an objective depending only on the PREDICTIVE VARIANCE cannot benefit from
    sequencing.  => rank by INFORMATION and you tend toward ONE list for everybody.
  - Sepliarskaia et al. (RecSys 2018): a STATIC questionnaire BEATS adaptive decision trees, and their own
    explanation is that the adaptive method *"optimizes a function that is different from the loss function,
    namely weighted generalized variance."*
  - Our probe ranked by the TASK LOSS (NDCG) and the order became strongly user-specific (+47% TAIL).

SO: rank the SAME 800 candidates, for the SAME clusters, on the SAME users, by an INFORMATION criterion
instead of by NDCG, and ask:
  (1) does the ENTROPY ranking agree ACROSS clusters more than the NDCG ranking does?  (theorem: yes)
  (2) does asking the ENTROPY-optimal question actually WIN on the task?                (theorem: no)

If entropy-ranking collapses toward one list AND underperforms, we have an EMPIRICAL demonstration of the
boundary the whole thesis turns on -- and it is Sepliarskaia's published mechanism, in our own data.

CRITERIA COMPARED (all computed WITHIN a cluster, on its SELECTION half; evaluated on the DISJOINT half):
  NDCG       : mean TAIL NDCG@10 after folding the answer          <- the TASK LOSS (what the probe used)
  ENTROPY    : H(answer | cluster, q) over {refuse, hated, meh, liked, loved}   <- an INFORMATION surrogate
  VARIANCE   : Var(answer value | cluster, q), answered cells only <- the classic design surrogate
  ANSWERRATE : P(answerable | cluster, q)                          <- the "ask what they can answer" heuristic
HARD RULE #1: all users, all 800 candidates, all clusters. No sampling.
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from signed_latent import load_arena_base   # noqa
import arena_core as AC                     # noqa

RS = "C:/dev/phd/casper/.cache/rich_signal"
META = "C:/dev/phd/casper/data/movielens/.cache/ml25m/meta.npz"
TAG_Q = "C:/dev/phd/casper/.cache/instrument2/tag_questions.json"
TAG_MEMB = "C:/dev/phd/casper/.cache/instrument2/tag_membership.json"
OFF_ITEM = 1628
GENRES = ['action','adventure','animation','comedy','crime','documentary','drama','family','fantasy',
          'horror','musical','mystery','noir','romance','sci-fi','thriller','war','western']


def log(m):
    print(m, flush=True)


def main():
    base = load_arena_base(); ni = base["ni"]; head = base["headmask"]
    d = np.load(META)
    uu = d["uu"].astype(np.int64); ii = d["ii"].astype(np.int64); rr = d["rr"].astype(np.float64)
    o = np.argsort(uu, kind="stable"); uu, ii, rr = uu[o], ii[o], rr[o]
    bnd = np.searchsorted(uu, np.arange(uu[-1] + 2))

    U, K, V = [], [], []
    for tag in ("train", "val"):
        U.append(np.load(f"{RS}/mm_{tag}_uids.npy"))
        K.append(np.load(f"{RS}/mm_{tag}_know.npy")); V.append(np.load(f"{RS}/mm_{tag}_val.npy"))
    uids = np.concatenate(U); K = np.concatenate(K); V = np.concatenate(V)

    tq = json.load(open(TAG_Q))["tags"]; memb = json.load(open(TAG_MEMB))["membership"]
    gmask = np.zeros((len(GENRES), ni), bool)
    for gi, g in enumerate(GENRES):
        tid = next(t["tagId"] for t in tq if str(t.get("tag")).lower() == g)
        for j in memb.get(str(tid), []):
            if 0 <= int(j) < ni:
                gmask[gi, int(j)] = True

    keep = []
    for r, u in enumerate(uids):
        a, b = bnd[int(u)], bnd[int(u) + 1]
        its, rat = ii[a:b], rr[a:b]
        if len(its) < 8:
            continue
        ru = np.random.default_rng(AC.SEED * 1_000_003 + int(u))
        perm = ru.permutation(len(its)); h = len(its) // 2
        ki = its[perm[:h]]; hi, hr = its[perm[h:]], rat[perm[h:]]
        hl = hi[hr >= 4]
        if len(ki) < 4 or len(hl) == 0 or len(hl[~head[hl]]) == 0:
            continue
        keep.append(r)
    keep = np.array(keep, np.int64)
    log(f"[e-vs-n] usable users = {len(keep)}  (must match the probe's 150,239)")

    # the probe's cached per-user TAIL NDCG for all 800 candidates, and the same clusters/halves
    ndT = np.load(".cache/probe2_ndT.npy").astype(np.float32)
    fav = np.load(".cache/probe2_fav.npy"); half = np.load(".cache/probe2_half.npy")
    assert len(fav) == len(keep) == ndT.shape[0], (len(fav), len(keep), ndT.shape)

    KI = K[keep][:, OFF_ITEM:]; VI = V[keep][:, OFF_ITEM:]          # (N, 800) know / val for the item bank
    ANS = (KI >= 1) & (VI >= 0)
    S = np.where(half)[0]; E = np.where(~half)[0]
    nb = KI.shape[1]

    def criteria(rows):
        """All four rankings, computed on the SAME users (a cluster's selection half)."""
        a = ANS[rows]; v = VI[rows]
        # 5-outcome answer distribution per question: refuse + hated/meh/liked/loved
        P = np.zeros((5, nb))
        P[0] = (~a).mean(0)
        for L in range(4):
            P[L + 1] = (a & (v == L)).mean(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            H = -np.where(P > 0, P * np.log(P), 0.0).sum(0)          # ENTROPY of the answer
        cnt = a.sum(0).clip(1)
        mu = np.where(a, v, 0).sum(0) / cnt
        var = (np.where(a, (v - mu) ** 2, 0).sum(0)) / cnt           # VARIANCE of the answer value
        return {"NDCG": ndT[rows].mean(0), "ENTROPY": H, "VARIANCE": var, "ANSWERRATE": a.mean(0)}

    names = ["NDCG", "ENTROPY", "VARIANCE", "ANSWERRATE"]
    big = [c for c in range(18) if (fav[S] == c).sum() >= 50 and (fav[E] == c).sum() >= 50]

    # ---- (1) do the criteria AGREE ACROSS CLUSTERS?  (theorem: entropy should agree far more than NDCG) ----
    from itertools import combinations
    sel = {c: criteria(S[fav[S] == c]) for c in big}
    log("\n(1) DO THE CLUSTERS WANT THE SAME QUESTION ORDER UNDER EACH CRITERION?")
    log(f"    {'criterion':<12}{'distinct top-1 across clusters':>32}{'mean top-10 overlap':>22}{'mean Spearman':>16}")
    log("    " + "-" * 82)
    for nm in names:
        tops = [int(np.argmax(sel[c][nm])) for c in big]
        ovs, sps = [], []
        for a, b in combinations(big, 2):
            ra = np.argsort(-sel[a][nm]); rb = np.argsort(-sel[b][nm])
            ovs.append(len(set(ra[:10]) & set(rb[:10])) / 10.0)
            pa = np.empty(nb); pa[ra] = np.arange(nb)
            pb = np.empty(nb); pb[rb] = np.arange(nb)
            sps.append(np.corrcoef(pa, pb)[0, 1])
        log(f"    {nm:<12}{len(set(tops)):>10} / {len(big):<19}{np.mean(ovs):>22.3f}{np.mean(sps):>16.3f}")
    log("    (top-10 overlap ~1.0 and Spearman ~1.0 => ONE LIST FOR EVERYBODY)")

    # ---- (2) does the criterion's chosen question actually WIN on the task? (evaluated out-of-sample) ----
    log("\n(2) OUT-OF-SAMPLE TAIL NDCG@10 OF THE QUESTION EACH CRITERION PICKS")
    log("    (chosen on the cluster's SELECTION half, scored on its DISJOINT EVAL half)")
    gq = int(np.argmax(ndT[S].mean(0)))
    static = float(ndT[E, gq].mean())
    log(f"    {'criterion':<12}{'TAIL':>9}{'vs static':>12}   per-cluster picks")
    log("    " + "-" * 82)
    rows_out = {}
    for nm in names:
        num = den = 0.0
        for c in big:
            ec = E[fav[E] == c]
            q = int(np.argmax(sel[c][nm]))
            num += float(ndT[ec, q].mean()) * len(ec); den += len(ec)
        val = num / den
        rows_out[nm] = val
        log(f"    {nm:<12}{val:>9.4f}{val - static:>+12.4f}")
    log(f"    {'STATIC':<12}{static:>9.4f}{0.0:>+12.4f}   (one globally-best question for everybody)")
    log("")
    log(f"  >>> TASK-LOSS ranking (NDCG)      : {rows_out['NDCG']:.4f}  ({rows_out['NDCG']-static:+.4f})")
    log(f"  >>> INFORMATION ranking (ENTROPY) : {rows_out['ENTROPY']:.4f}  ({rows_out['ENTROPY']-static:+.4f})")
    log("  If ENTROPY <= STATIC while NDCG >> STATIC, then Sepliarskaia's mechanism is demonstrated in our own")
    log("  data: an adaptive policy that optimises an INFORMATION SURROGATE loses to a static one; the prize")
    log("  only appears when you rank by the TASK LOSS. That is the boundary the whole thesis turns on.")


if __name__ == "__main__":
    main()
