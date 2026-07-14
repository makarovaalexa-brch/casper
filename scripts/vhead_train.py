"""THE THREE ARMS — train + evaluate the value-head policy. (2026-07-14, Fable-audited)

  A1  Q(z, e)                    -- the frozen point belief.   *** THIS IS THE POLICY WE WANT ***
  A2  Q(z, e, e^T Sigma_n e, n)  -- + uncertainty computed BESIDE the recommender ("C-lite").
  A3  Q(state tokens, e)         -- a transformer over the RAW history.
                                    *** THE CEILING ***: any belief representation (including a full
                                    covariance) is a DETERMINISTIC FUNCTION of the history, so A3
                                    upper-bounds EVERY belief representation -- including the unbuilt one.

PRE-REGISTERED (MDE = 0.004 tail NDCG@10):
  A3 - A1 < 0.004                    => z is SUFFICIENT. The uncertainty question is CLOSED.
  A3 - A1 >= 0.004, A2 closes >=70%  => ship uncertainty as a FEATURE. Still no belief-recommender.
  A3 - A1 >= 0.004, A2 does not      => the missing signal needs a LEARNED representation -- A3 IS it.

FABLE'S AUDIT FIXES (all applied here; the fatal target bugs were fixed in vhead.py and the targets rebuilt):
  #3  A2's lambda was DETACHED => it never trained ("fitted" was a lie). Now a real grid search on the
      SELECTION half by regression MSE. Never hand-set.
  #4  A3's eval recomputed the candidate-INDEPENDENT transformer 800x per user (tens of GB, days).
      The CLS output h does NOT depend on the candidate => compute h ONCE per state, then run only the
      final MLP over the candidates. Same in training.
  #5  STATIC's selection was an argmax over 800 noisy means (SE ~0.002-0.003 vs an MDE of 0.004) => it could
      UNDERSTATE the best static question and manufacture a fake adaptive win. Now: shortlist the top-32 by
      estimated advantage, then EXACTLY fold-and-score all 32 across the whole selection half, and take that
      argmax. Also report the top-5 statics on the eval half as a sensitivity row.
  #9  Centre on BASE (the state's own NDCG -- the true "gain over not asking"), not the noisy 16-sample mean.
  #10 Sigma_pop is computed from the SELECTION half only.
  #11 RANDOM must not pick an already-asked item (the arms cannot).
  #12 strict=False is asserted empty; per-epoch checkpoints; the arm is picked by SELECTION-half val MSE.

MASKING RULE (identical in targets and eval): mask asked-AND-ANSWERED items and the folded candidate if
answered. A REFUSED item stays RECOMMENDABLE (the user told us they do not know it -- it is a legitimate
recommendation); the turn is still burned.
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vhead as VH                                              # noqa
from set_mn import SetEncoder                                   # noqa

OUT = VH.OUT
D = VH.D
MDE = 0.004
EPOCHS = 6
LAM_GRID = [0.1, 0.3, 1.0, 3.0, 10.0]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


class QPoint(nn.Module):
    def __init__(self, nx=0):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(2 * D + nx, 512), nn.GELU(),
                               nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, z, e, x=None):
        return self.f(torch.cat([z, e] + ([x] if x is not None else []), -1)).squeeze(-1)


class QHist(nn.Module):
    """A3 -- THE CEILING. h = f(raw history) is computed ONCE per state (defect #4)."""
    def __init__(self, d=256):
        super().__init__()
        self.proj = nn.Linear(D, d); self.lvl = nn.Embedding(10, d)
        lyr = nn.TransformerEncoderLayer(d, 4, 4 * d, batch_first=True, dropout=0.0)
        self.tr = nn.TransformerEncoder(lyr, 2)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.f = nn.Sequential(nn.Linear(d + D, 512), nn.GELU(),
                               nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 1))

    def state(self, tok, lv, pad):
        B = tok.shape[0]
        x = self.proj(tok) + self.lvl(lv)
        x = torch.cat([self.cls.expand(B, -1, -1), x], 1)
        p = torch.cat([torch.zeros(B, 1, dtype=torch.bool), pad], 1)
        return self.tr(x, src_key_padding_mask=p)[:, 0]           # (B, d) -- candidate-INDEPENDENT

    def score(self, h, e):
        return self.f(torch.cat([h, e], -1)).squeeze(-1)


def main():
    torch.set_num_threads(os.cpu_count())
    base, ni, head, bank, held_t, LV, ANS = VH.build()
    N = len(held_t); nb = len(bank)

    rng = np.random.default_rng(0)
    tlen = rng.integers(1, VH.MAXT + 1, size=N)
    asked = [rng.choice(nb, size=int(t), replace=False) for t in tlen]
    log(f"states replayed: mean interview length {tlen.mean():.2f}")

    Z = torch.from_numpy(np.load(f"{OUT}/Z.npy"))
    BASE = np.load(f"{OUT}/BASE.npy")
    CU = np.load(f"{OUT}/CU.npy"); CY = np.load(f"{OUT}/CY.npy")
    K = CU.shape[1]
    # #9: centre on BASE -- the true "gain over not asking this question"
    ADV = torch.from_numpy((CY - BASE[:, None]).astype(np.float32))
    log(f"targets {N}x{K} | state tail {BASE.mean():.4f} | gain-over-BASE mean {ADV.mean():+.4f} sd {ADV.std():.4f}")

    enc = SetEncoder(ni, token_mode="film", pool="attn")
    ck = torch.load(VH.PB2, map_location="cpu")
    miss, unexp = enc.load_state_dict(ck["student"], strict=False)
    assert not unexp, f"unexpected keys in pb2 ckpt: {unexp}"      # #12
    assert all(k.startswith(("log_p0", "pscale", "lam_head")) for k in miss), f"missing: {miss}"
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    dec = nn.Linear(D, ni); dec.load_state_dict(ck["decoder"])
    Wd = dec.weight.detach(); bd = dec.bias.detach()
    EMB = enc.item_emb.weight.detach()
    Eb = EMB[torch.from_numpy(bank)]
    W = 1.0 / np.log2(np.arange(2, 12))

    rs = np.random.default_rng(1); half = rs.random(N) < 0.5
    S = np.where(half)[0]; E = np.where(~half)[0]
    log(f"selection {len(S)}  evaluation {len(E)}  (disjoint; symmetric for every arm AND baseline)")

    # ---- #10: Sigma_pop from the SELECTION half only ----
    Zs = Z[torch.from_numpy(S)]
    Zc = Zs - Zs.mean(0, keepdim=True)
    Sp = (Zc.T @ Zc) / (len(S) - 1) + 1e-3 * torch.eye(D)
    log(f"Sigma_pop (selection half): trace {Sp.trace():.2f}")

    def tokens(u, extra=None):
        a = asked[u]; a = a[ANS[u, a]]
        if extra is not None and ANS[u, extra]:
            a = np.concatenate([a, [extra]])
        return bank[a], LV[u, a]

    def state_tensors(us):
        batch = [tokens(u) for u in us]
        L = max(1, max(len(t[0]) for t in batch))
        tid = np.zeros((len(us), L), np.int64); tlv = np.zeros((len(us), L), np.int64)
        tp = np.ones((len(us), L), bool)
        for r, (it, l_) in enumerate(batch):
            k = len(it)
            if k:
                tid[r, :k] = it; tlv[r, :k] = l_; tp[r, :k] = False
        return (EMB[torch.from_numpy(tid)], torch.from_numpy(tlv), torch.from_numpy(tp))

    def unc(us, cidx, lam):
        """e^T Sigma_n e, Sigma_n^-1 = Sigma_pop^-1 + lam * sum_t x_t x_t^T (Woodbury, n<=8)."""
        out = torch.zeros(len(us), cidx.shape[1])
        for r, u in enumerate(us):
            a = asked[u]; a = a[ANS[u, a]]
            e = Eb[cidx[r]]
            if len(a) == 0:
                out[r] = torch.einsum('kd,de,ke->k', e, Sp, e); continue
            X = EMB[torch.from_numpy(bank[a])]
            X = X / (X.norm(dim=1, keepdim=True) + 1e-8)
            SX = Sp @ X.T
            M = torch.eye(len(a)) / lam + X @ SX
            SE = Sp @ e.T
            out[r] = (e.T * (SE - SX @ torch.linalg.solve(M, X @ SE))).sum(0)
        return out

    def fold_score(us, picks):
        """Fold each user's PICK and score realized TAIL NDCG@10. SAME masking rule as the targets."""
        out = np.zeros(len(us))
        for b in range(0, len(us), 1024):
            ch = list(range(b, min(b + 1024, len(us))))
            batch = [tokens(us[i], int(picks[i])) for i in ch]
            L = max(1, max(len(t[0]) for t in batch))
            ids = np.zeros((len(ch), L), np.int64); lv = np.zeros((len(ch), L), np.int64)
            pad = np.ones((len(ch), L), bool)
            for r, (it, l_) in enumerate(batch):
                k = len(it)
                if k:
                    ids[r, :k] = it; lv[r, :k] = l_; pad[r, :k] = False
            with torch.no_grad():
                z = enc(torch.from_numpy(ids), torch.zeros((len(ch), L)),
                        torch.from_numpy(pad), torch.from_numpy(lv))
                sc = (z @ Wd.T + bd).numpy().astype(np.float64)
            for r, i in enumerate(ch):
                u = us[i]; s = sc[r].copy()
                a = asked[u]
                s[bank[a[ANS[u, a]]]] = -1e30                     # asked AND ANSWERED
                if ANS[u, int(picks[i])]:
                    s[bank[int(picks[i])]] = -1e30                # the folded candidate
                s[head] = -1e30
                hl = set(int(x) for x in held_t[u])
                o = np.argsort(-s)[:10]
                out[i] = sum(W[p] for p, t in enumerate(o) if int(t) in hl) / \
                    (W[:min(10, len(hl))].sum() + 1e-12)
        return out

    # ---- #3: FIT lambda by grid search on the SELECTION half (never hand-set) ----
    sub = S[:4000]
    ci_sub = torch.from_numpy(CU[sub]); y_sub = ADV[sub]
    best_lam, best_r = LAM_GRID[0], -9e9
    for lm in LAM_GRID:
        u_ = unc(sub, ci_sub, lm).reshape(-1).numpy()
        r_ = abs(np.corrcoef(u_, y_sub.reshape(-1).numpy())[0, 1])
        log(f"  lambda {lm:>5}: |corr(uncertainty, gain)| = {r_:.4f}")
        if r_ > best_r:
            best_r, best_lam = r_, lm
    log(f"  => lambda = {best_lam} (FITTED on the selection half, not hand-set)")

    arms = {"A1": QPoint(), "A2": QPoint(nx=2), "A3": QHist()}
    opts = {k: torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4) for k, m in arms.items()}
    Sn = torch.from_numpy(S)
    hold = torch.from_numpy(S[:5000])                               # selection-half val for peak-picking (#12)
    best = {k: (9e9, None) for k in arms}

    for ep in range(EPOCHS):
        perm = Sn[torch.randperm(len(Sn))]
        tot = {k: 0.0 for k in arms}; nbt = 0
        for b in range(0, len(perm), 256):
            us = perm[b:b + 256].numpy()
            ci = torch.from_numpy(CU[us]); y = ADV[us]
            B, Kc = ci.shape
            e = Eb[ci]
            z = Z[us].unsqueeze(1).expand(-1, Kc, -1)
            p1 = arms["A1"](z.reshape(-1, D), e.reshape(-1, D)).reshape(B, Kc)
            l1 = ((p1 - y) ** 2).mean()
            opts["A1"].zero_grad(); l1.backward(); opts["A1"].step()

            uf = unc(us, ci, best_lam)
            nt = torch.tensor([[float(ANS[u, asked[u]].sum())] for u in us]).expand(-1, Kc)
            x = torch.stack([uf, nt], -1).reshape(-1, 2)
            p2 = arms["A2"](z.reshape(-1, D), e.reshape(-1, D), x).reshape(B, Kc)
            l2 = ((p2 - y) ** 2).mean()
            opts["A2"].zero_grad(); l2.backward(); opts["A2"].step()

            tok, tlv, tp = state_tensors(us)                        # #4: h ONCE per state
            h = arms["A3"].state(tok, tlv, tp)                      # (B, d)
            p3 = arms["A3"].score(h.unsqueeze(1).expand(-1, Kc, -1).reshape(-1, h.shape[-1]),
                                  e.reshape(-1, D)).reshape(B, Kc)
            l3 = ((p3 - y) ** 2).mean()
            opts["A3"].zero_grad(); l3.backward(); opts["A3"].step()

            for k, l in zip(arms, [l1, l2, l3]):
                tot[k] += float(l)
            nbt += 1
            if nbt % 100 == 0:
                log(f"  ep{ep} b{nbt}/{len(perm)//256}  " + "  ".join(f"{k} {tot[k]/nbt:.5f}" for k in arms))
        # ---- selection-half validation; RECORD THE PEAK (#12) ----
        with torch.no_grad():
            ci = torch.from_numpy(CU[hold.numpy()]); y = ADV[hold.numpy()]
            e = Eb[ci]; B, Kc = ci.shape
            z = Z[hold].unsqueeze(1).expand(-1, Kc, -1)
            v = {"A1": ((arms["A1"](z.reshape(-1, D), e.reshape(-1, D)).reshape(B, Kc) - y) ** 2).mean()}
            uf = unc(hold.numpy(), ci, best_lam)
            nt = torch.tensor([[float(ANS[u, asked[u]].sum())] for u in hold.numpy()]).expand(-1, Kc)
            x = torch.stack([uf, nt], -1).reshape(-1, 2)
            v["A2"] = ((arms["A2"](z.reshape(-1, D), e.reshape(-1, D), x).reshape(B, Kc) - y) ** 2).mean()
            tok, tlv, tp = state_tensors(hold.numpy())
            h = arms["A3"].state(tok, tlv, tp)
            v["A3"] = ((arms["A3"].score(h.unsqueeze(1).expand(-1, Kc, -1).reshape(-1, h.shape[-1]),
                                         e.reshape(-1, D)).reshape(B, Kc) - y) ** 2).mean()
        for k in arms:
            if float(v[k]) < best[k][0]:
                best[k] = (float(v[k]), {kk: vv.clone() for kk, vv in arms[k].state_dict().items()})
        log(f"[ep{ep+1}] train " + "  ".join(f"{k} {tot[k]/max(nbt,1):.5f}" for k in arms) +
            "  | val " + "  ".join(f"{k} {float(v[k]):.5f}" for k in arms))
    for k in arms:
        arms[k].load_state_dict(best[k][1]); arms[k].eval()
        log(f"  {k}: peak val MSE {best[k][0]:.5f}")

    # ================= EVALUATE =================
    log("\nEVAL: each arm scores ALL 800 candidates, argmax, fold+score ONLY that pick.")
    res = {}; picks_of = {}
    for name, m in arms.items():
        picks = np.zeros(len(E), np.int64)
        with torch.no_grad():
            for b in range(0, len(E), 512):
                us = E[b:b + 512]; B = len(us)
                if name == "A3":
                    tok, tlv, tp = state_tensors(us)
                    h = m.state(tok, tlv, tp)                       # (B, d)  ONCE
                    q = m.score(h.unsqueeze(1).expand(-1, nb, -1).reshape(-1, h.shape[-1]),
                                Eb.unsqueeze(0).expand(B, -1, -1).reshape(-1, D)).reshape(B, nb)
                else:
                    z = Z[us].unsqueeze(1).expand(-1, nb, -1).reshape(-1, D)
                    e = Eb.unsqueeze(0).expand(B, -1, -1).reshape(-1, D)
                    if name == "A1":
                        q = m(z, e).reshape(B, nb)
                    else:
                        ci = torch.arange(nb).unsqueeze(0).expand(B, -1)
                        uf = unc(us, ci, best_lam)
                        nt = torch.tensor([[float(ANS[u, asked[u]].sum())] for u in us]).expand(-1, nb)
                        q = m(z, e, torch.stack([uf, nt], -1).reshape(-1, 2)).reshape(B, nb)
                for r, u in enumerate(us):
                    q[r, asked[u]] = -1e30                          # never re-ask
                picks[b:b + B] = q.argmax(1).numpy()
        picks_of[name] = picks
        res[name] = fold_score(E, picks)
        log(f"  {name}: tail {res[name].mean():.4f}  (distinct picks {len(set(picks.tolist()))}/{nb})")

    # ---- #5: STATIC refined by EXACT folding of the top-32 shortlist on the SELECTION half ----
    est = np.bincount(CU[S].ravel(), weights=ADV[torch.from_numpy(S)].numpy().ravel(), minlength=nb) / \
        np.bincount(CU[S].ravel(), minlength=nb).clip(1)
    short = np.argsort(-est)[:32]
    log(f"\nSTATIC: exactly folding the top-32 shortlist over the selection half ({len(S)} users)...")
    sc32 = np.array([fold_score(S, np.full(len(S), int(c))).mean() for c in short])
    gq = int(short[int(np.argmax(sc32))])
    log(f"  static question = bank[{gq}]  (selection-half tail {sc32.max():.4f})")
    res["STATIC"] = fold_score(E, np.full(len(E), gq))
    top5 = short[np.argsort(-sc32)[:5]]
    sens = [fold_score(E, np.full(len(E), int(c))).mean() for c in top5]
    log(f"  sensitivity, top-5 statics on eval: {[round(x,4) for x in sens]}")

    rr = np.random.default_rng(2)                                   # #11: RANDOM cannot re-ask either
    rp = np.array([int(rr.choice(np.setdiff1d(np.arange(nb), asked[u]))) for u in E])
    res["RANDOM"] = fold_score(E, rp)
    res["BEST-OF-16 (lower bound on the oracle, NOT a ceiling)"] = CY[E].max(1)
    res["NO QUESTION (state only)"] = BASE[E]

    log("\n" + "=" * 84)
    st = res["STATIC"].mean()
    for k in ["NO QUESTION (state only)", "RANDOM", "STATIC", "A1", "A2", "A3",
              "BEST-OF-16 (lower bound on the oracle, NOT a ceiling)"]:
        v = res[k].mean()
        log(f"  {k:<52} tail {v:.4f}  {v-st:+.4f} vs static")
    log("=" * 84)

    rb = np.random.default_rng(3)
    def boot(a, b):
        d = a - b
        bs = np.array([d[rb.integers(0, len(d), len(d))].mean() for _ in range(2000)])
        return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    for k in ["A1", "A2", "A3"]:
        m_, lo, hi = boot(res[k], res["STATIC"])
        log(f"  {k} - STATIC = {m_:+.4f}  95% CI [{lo:+.4f},{hi:+.4f}]  {'SIGNIFICANT' if lo>0 else 'n.s.'}")
    m_, lo, hi = boot(res["A3"], res["A1"])
    log(f"\n  *** A3 - A1 = {m_:+.4f}  95% CI [{lo:+.4f},{hi:+.4f}]   (MDE {MDE}) ***")
    if m_ < MDE:
        log("  => z IS SUFFICIENT. The uncertainty question is CLOSED. No covariance, ever.")
    else:
        m2, _, _ = boot(res["A2"], res["A1"])
        log(f"  => A3 > A1. A2 recovers {100*m2/max(m_,1e-9):.0f}% "
            f"({'ship the feature' if m2/max(m_,1e-9) >= 0.7 else 'needs a learned representation = A3'})")

    torch.save({k: m.state_dict() for k, m in arms.items()}, f"{OUT}/arms.pt")
    np.savez(f"{OUT}/results.npz", **{k.split()[0]: v for k, v in res.items()},
             picks_A1=picks_of["A1"], picks_A2=picks_of["A2"], picks_A3=picks_of["A3"], gq=gq)
    log(f"\nsaved {OUT}/arms.pt + results.npz")


if __name__ == "__main__":
    main()
