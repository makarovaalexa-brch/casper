"""THE THREE-ARM TEST — does the policy need an explicit uncertainty, or is the point belief enough?
(2026-07-14. Fable-designed, author-approved.)

THE QUESTION. Our belief z is trained ONLY to rank items, then FROZEN. A regression Q(z,e) converges to
E[value | z, e]. If two answer-histories with the same z carry different residual uncertainty, Q learns their
AVERAGE and can never split them -- no amount of data fixes an invariance in the input. So: has the frozen z
LOST something the policy needs?

THREE ARMS. Same targets, same users, same split. They differ ONLY in what the policy can SEE:
  A1  Q(z, e_c)                      -- the frozen point belief.              (this IS the policy we want)
  A2  Q(z, e_c, e^T Sigma_n e, ...)  -- + uncertainty, computed BESIDE the recommender ("C-lite"):
                                        Sigma_n^-1 = Sigma_pop^-1 + sum_t lam * x_t x_t^T,
                                        Sigma_pop = empirical covariance of z across users (the real prior).
  A3  Q(history tokens, e_c)         -- a small set encoder over the RAW answer tokens.
                                        *** THE CEILING ***: any belief representation (including a full
                                        covariance) is a DETERMINISTIC FUNCTION of the history, so A3
                                        upper-bounds EVERY belief representation, including the one we did
                                        not build.

DECISION RULE (pre-registered, MDE = 0.004 tail NDCG@10):
  A3 - A1 < 0.004                      => z is SUFFICIENT. The uncertainty question is CLOSED.
  A3 - A1 >= 0.004, A2 closes >=70%    => ship the uncertainty as a FEATURE (A2). Still no belief-recommender.
  A3 - A1 >= 0.004, A2 does not        => the missing signal needs a LEARNED representation -- and A3 IS it,
                                          already trained. Still policy-side.
In every branch the answer stays POLICY-SIDE. (That is why the belief-recommender line is closed.)

CRITICAL DESIGN POINTS (Fable's three fixes -- the third is the one that makes the test valid at all):
  1. Predict the ADVANTAGE (per-state centred), not the raw NDCG. Raw score is dominated by "is this user
     easy or hard", which z explains trivially and the policy CANNOT ACT ON (argmax is invariant to a
     per-state constant). Centring is what makes the target the thing the policy actually chooses on.
  2. A3 is the CEILING arm. Without it, a weak A1 is uninterpretable.
  3. *** MID-INTERVIEW STATES, NOT TURN 1. *** At turn 1 every user has the identical (empty) history and
     therefore identical uncertainty -- a turn-1 test is STRUCTURALLY INCAPABLE of detecting the effect.
     States are sampled from interviews of length 1..8.

HONESTY:
  * REFUSALS BURN THE TURN. If the answerer refuses, the turn is spent and the belief does NOT move (no
    token). A candidate the user would refuse therefore has value = the state's own NDCG (zero gain).
  * All 150k labelled users. All 800 bank candidates at decision time. No sampling of users or candidates at
    EVAL. (Candidates ARE sampled when building TRAINING targets -- that is minibatch SGD over a huge
    (state, candidate) space, not a data reduction.)
  * TAIL NDCG@10 is the primary metric (the headline metric of Papers B/C).
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from set_mn import SetEncoder, sv_to_level                                  # noqa
from signed_latent import load_arena_base, scale_rating, ndcg10             # noqa
import arena_core as AC                                                     # noqa

PB2 = "C:/dev/phd/casper/.cache/set_mn/pb2_best.pt"
RS = "C:/dev/phd/casper/.cache/rich_signal"
META = "C:/dev/phd/casper/data/movielens/.cache/ml25m/meta.npz"
ITEM_LISTS = "C:/dev/phd/casper/.cache/instrument2/item_lists.json"
OFF_ITEM = 1628
OUT = "C:/dev/phd/casper/.cache/vhead"
os.makedirs(OUT, exist_ok=True)
D = 512
KCAND = 16          # candidates sampled per state when BUILDING TRAINING TARGETS (minibatch SGD over the
                    # (state, candidate) product; DECLARED to the author. All 800 would be ~107h.
                    # Coverage: 150,238 x 16 / 800 ~= 3,005 pairs per candidate (~1,500 in the selection half).
MAXT = 8            # interview length 1..8 (mid-interview states -- Fable fix #3)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------- data
def build():
    base = load_arena_base(); ni = base["ni"]; head = base["headmask"]
    d = np.load(META)
    uu = d["uu"].astype(np.int64); ii = d["ii"].astype(np.int64); rr = d["rr"].astype(np.float64)
    o = np.argsort(uu, kind="stable"); uu, ii, rr = uu[o], ii[o], rr[o]
    bnd = np.searchsorted(uu, np.arange(uu[-1] + 2))
    bank = np.array([int(x) for x in json.load(open(ITEM_LISTS))["lists"]["top800"]["ids"]], np.int64)

    U, K, V, CR = [], [], [], []
    for tag in ("train", "val"):
        U.append(np.load(f"{RS}/mm_{tag}_uids.npy"))
        K.append(np.load(f"{RS}/mm_{tag}_know.npy")); V.append(np.load(f"{RS}/mm_{tag}_val.npy"))
        CR.append(np.load(f"{RS}/mm_{tag}_crval.npy"))
    uids = np.concatenate(U); K = np.concatenate(K)[:, OFF_ITEM:]; V = np.concatenate(V)[:, OFF_ITEM:]
    CR = np.concatenate(CR)
    log(f"labelled users {len(uids)}   bank {len(bank)}")

    rows, held_t, kmean = [], [], []
    for r, u in enumerate(uids):
        a, b = bnd[int(u)], bnd[int(u) + 1]
        its, rat = ii[a:b], rr[a:b]
        if len(its) < 8:
            continue
        ru = np.random.default_rng(AC.SEED * 1_000_003 + int(u))
        p = ru.permutation(len(its)); h = len(its) // 2
        ki, kr = its[p[:h]], rat[p[:h]]
        hi, hr = its[p[h:]], rat[p[h:]]
        hl = hi[hr >= 4]
        if len(ki) < 4 or len(hl) == 0:
            continue
        hlt = hl[~head[hl]]
        if len(hlt) == 0:
            continue
        rows.append(r); held_t.append(hlt); kmean.append(float(kr.mean()))
    rows = np.array(rows, np.int64); kmean = np.array(kmean)
    K, V, CR = K[rows], V[rows], CR[rows]
    log(f"usable users {len(rows)}")

    # ordinal -> star, DERIVED FROM THE DATA (crval is NaN for unrated bank items)
    star = CR + kmean[:, None]
    L2S = np.zeros(4)
    for L in range(4):
        m = np.isfinite(star) & (V == L)
        L2S[L] = float(np.nanmean(star[m])) if m.any() else 3.0
    st = np.where(np.isfinite(star), star, L2S[np.clip(V, 0, 3)])
    st = np.clip(np.rint(st * 2) / 2.0, 0.5, 5.0)
    LV = np.clip(np.rint(st * 2).astype(np.int64) - 1, 0, 9)      # (N,800) item star level
    ANS = (K >= 1) & (V >= 0)                                     # answered? else the turn is BURNED
    log(f"ordinal->star (from data): {dict(zip(['hated','meh','liked','loved'], np.round(L2S,3)))}")
    log(f"answer rate on the item bank: {ANS.mean():.4f}")
    return base, ni, head, bank, held_t, LV, ANS


def main():
    torch.set_num_threads(os.cpu_count())
    base, ni, head, bank, held_t, LV, ANS = build()
    N = len(held_t); nb = len(bank)

    enc = SetEncoder(ni, token_mode="film", pool="attn")
    ck = torch.load(PB2, map_location="cpu")
    enc.load_state_dict(ck["student"], strict=False); enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    dec = nn.Linear(D, ni); dec.load_state_dict(ck["decoder"])
    Wd = dec.weight.detach(); bd = dec.bias.detach()
    log(f"pb2 frozen (full {ck['full']:.4f} tail {ck['tail']:.4f})")

    rng = np.random.default_rng(0)
    # ---- mid-interview STATES (Fable fix #3): 1..8 asked bank items per user ----
    tlen = rng.integers(1, MAXT + 1, size=N)
    asked = [rng.choice(nb, size=int(t), replace=False) for t in tlen]

    W = 1.0 / np.log2(np.arange(2, 12))

    def tokens_of(u, extra=None):
        """The state's ANSWERED items (+ optionally one candidate). A REFUSED ask BURNS THE TURN: it
        consumes a question but contributes NO token -- the belief does not move."""
        a = asked[u]
        a = a[ANS[u, a]]
        if extra is not None and ANS[u, extra]:
            a = np.concatenate([a, [extra]])
        return bank[a], LV[u, a]

    def encode(batch):
        L = max(1, max(len(t[0]) for t in batch)); B = len(batch)
        ids = np.zeros((B, L), np.int64); lv = np.zeros((B, L), np.int64)
        pad = np.ones((B, L), bool)
        for r, (it, l_) in enumerate(batch):
            k = len(it)
            if k:
                ids[r, :k] = it; lv[r, :k] = l_; pad[r, :k] = False
        with torch.no_grad():
            return enc(torch.from_numpy(ids), torch.zeros((B, L)), torch.from_numpy(pad),
                       torch.from_numpy(lv))

    def tail_ndcg(z, us, picks=None):
        """TAIL NDCG@10 of the belief z.
        MASKING RULE (identical in targets and in eval -- Fable defect #1 was that they DIFFERED):
          * mask every ASKED-AND-ANSWERED item (we now know the user's opinion -> not a recommendation), and
          * mask the FOLDED CANDIDATE if it was answered.
            Without this the target is CIRCULAR: fold 'user loves Blade Runner' -> the model ranks Blade
            Runner first -> a large fake DCG bonus for asking about it.
          * a REFUSED item stays RECOMMENDABLE: the user has told us they do not know it, which makes it a
            legitimate (arguably ideal) recommendation. The turn is still burned."""
        sc = (z @ Wd.T + bd).numpy().astype(np.float64)
        out = np.zeros(len(us))
        for r, u in enumerate(us):
            s = sc[r].copy()
            a = asked[u]
            s[bank[a[ANS[u, a]]]] = -1e30                 # asked AND ANSWERED -> known -> not recommendable
            if picks is not None and ANS[u, picks[r]]:
                s[bank[picks[r]]] = -1e30                 # *** THE FOLDED CANDIDATE. defect #1. ***
            s[head] = -1e30                               # TAIL: head items masked out
            hl = set(int(x) for x in held_t[u])
            o = np.argsort(-s)[:10]
            dcg = sum(W[p] for p, t in enumerate(o) if int(t) in hl)
            out[r] = dcg / (W[:min(10, len(hl))].sum() + 1e-12)
        return out

    # ---------- STATE beliefs z (one encode per user) ----------
    # RESUME: the encoder pass is unaffected by the target bug, so Z is reusable. BASE is RECOMPUTED
    # (it depends on the masking rule, which changed).
    if os.path.exists(f"{OUT}/Z.npy"):
        Z = np.load(f"{OUT}/Z.npy")
        assert Z.shape == (N, D), f"stale Z {Z.shape} vs {(N, D)}"
        log("Z reused from disk (encoder pass unaffected by the target bug)")
    else:
        log("encoding states ...")
        Z = np.zeros((N, D), np.float32)
        for b in range(0, N, 2048):
            us = list(range(b, min(b + 2048, N)))
            Z[us] = encode([tokens_of(u) for u in us]).numpy()
            if b % 20480 == 0:
                log(f"  states {b}/{N}")
        np.save(f"{OUT}/Z.npy", Z)
    BASE = np.zeros(N)
    for b in range(0, N, 4096):
        us = list(range(b, min(b + 4096, N)))
        BASE[us] = tail_ndcg(torch.from_numpy(Z[us]), us)          # under the CORRECTED masking rule
    np.save(f"{OUT}/BASE.npy", BASE)
    log(f"state tail-NDCG (before the next question): {BASE.mean():.4f}")

    # ---------- TRAINING TARGETS: realized tail NDCG after folding a sampled candidate ----------
    log(f"building targets: {N} states x {KCAND} sampled candidates ...")
    CU = np.zeros((N, KCAND), np.int64); CY = np.zeros((N, KCAND), np.float32)
    t0 = time.time()
    for b in range(0, N, 512):
        us = list(range(b, min(b + 512, N)))
        flat, owner, picks = [], [], []
        cand = np.zeros((len(us), KCAND), np.int64)
        for r, u in enumerate(us):
            # DEFECT #2: never sample a candidate the user has ALREADY been asked. Re-asking is an INVALID
            # ACTION (eval bans it), and appending its token twice would teach Q that re-asking sharpens the
            # belief. Excluding invalid actions is not a data reduction.
            pool = np.setdiff1d(np.arange(nb), asked[u], assume_unique=False)
            c = rng.choice(pool, size=KCAND, replace=False)
            cand[r] = c
            for j in range(KCAND):
                flat.append(tokens_of(u, int(c[j]))); owner.append(u); picks.append(int(c[j]))
        z = encode(flat)
        y = tail_ndcg(z, owner, np.array(picks))                   # <-- candidate MASKED (defect #1)
        CU[us] = cand; CY[us] = y.reshape(len(us), KCAND)
        if b % 10240 == 0 and b:
            el = time.time() - t0
            log(f"  targets {b}/{N}  {el/60:.0f}m elapsed, ETA {(el/b*(N-b))/60:.0f}m")
    np.save(f"{OUT}/CU.npy", CU); np.save(f"{OUT}/CY.npy", CY)
    log(f"targets done. mean realized {CY.mean():.4f} vs state base {BASE.mean():.4f}")


if __name__ == "__main__":
    main()
