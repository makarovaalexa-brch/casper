"""THE THREE ARMS — train + evaluate the value-head policy. (2026-07-14)

Reads the targets built by `vhead.py` and answers TWO questions at once:
  Q1 (the policy):     does a value head Q(z,e) trained on REALIZED tail NDCG BEAT a static questionnaire?
  Q2 (the uncertainty): does the policy need an explicit uncertainty, or is the frozen point belief enough?

  A1  Q(z, e)                    -- the frozen point belief.        *** THIS IS THE POLICY WE WANT ***
  A2  Q(z, e, e^T Sigma_n e, n)  -- + uncertainty computed BESIDE the recommender ("C-lite").
  A3  Q(state tokens, e)         -- a set encoder over the RAW history.
                                    *** THE CEILING ***: any belief representation (including a full
                                    covariance) is a DETERMINISTIC FUNCTION of the history, so A3
                                    upper-bounds EVERY belief representation -- including the unbuilt one.

PRE-REGISTERED DECISION RULE (MDE = 0.004 tail NDCG@10):
  A3 - A1 < 0.004                    => z is SUFFICIENT. The uncertainty question is CLOSED.
  A3 - A1 >= 0.004, A2 closes >=70%  => ship uncertainty as a FEATURE. Still no belief-recommender.
  A3 - A1 >= 0.004, A2 does not      => the missing signal needs a LEARNED representation -- A3 IS it.

TARGET = per-state-centred ADVANTAGE:  A(u,c) = Y(u,c) - mean_c' Y(u,c').
Raw realized NDCG is dominated by "is this user easy or hard", which z explains trivially and the policy
CANNOT ACT ON (argmax is invariant to a per-state constant). Regressing raw Y would give a high R^2 that
certifies nothing.

EVALUATION = a POLICY evaluation, not a regression score. Each arm scores ALL 800 candidates with its own
cheap head (z is fixed; the candidate embedding is a lookup -> free), takes its ARGMAX, and we FOLD AND SCORE
ONLY THAT PICK through the frozen recommender. Primary metric: realized TAIL NDCG@10, paired bootstrap.

DECLARED CHOICES (author informed; nothing silent):
  * TRAINING targets used KCAND=16 candidates/state (all 800 would be ~107h). EVALUATION uses ALL 800.
  * BEST-OF-SAMPLED is the best of those 16 => a LOWER BOUND on the true 800-candidate oracle. Labelled so.
  * Refusals BURN THE TURN (no token, no belief movement), in the state and in the fold of a pick.
  * Selection/evaluation on DISJOINT halves of users, symmetric for every arm AND every baseline.
  * All 150,239 users. No user dropped at eval. No candidate dropped at eval.
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vhead as VH                                              # noqa  (reuse build() + constants)
from set_mn import SetEncoder                                   # noqa

OUT = VH.OUT
D = VH.D
MDE = 0.004
EPOCHS = 6


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


class QPoint(nn.Module):
    def __init__(self, nx=0):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(2 * D + nx, 512), nn.GELU(),
                               nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, z, e, x=None):
        h = torch.cat([z, e] + ([x] if x is not None else []), -1)
        return self.f(h).squeeze(-1)


class QHist(nn.Module):
    """A3 — the ceiling: a transformer over the RAW state tokens + the candidate."""
    def __init__(self, d=256):
        super().__init__()
        self.proj = nn.Linear(D, d); self.lvl = nn.Embedding(10, d)
        lyr = nn.TransformerEncoderLayer(d, 4, 4 * d, batch_first=True, dropout=0.0)
        self.tr = nn.TransformerEncoder(lyr, 2)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.f = nn.Sequential(nn.Linear(d + D, 512), nn.GELU(),
                               nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, tok, lv, pad, e):
        B = tok.shape[0]
        x = self.proj(tok) + self.lvl(lv)
        x = torch.cat([self.cls.expand(B, -1, -1), x], 1)
        p = torch.cat([torch.zeros(B, 1, dtype=torch.bool), pad], 1)
        h = self.tr(x, src_key_padding_mask=p)[:, 0]
        return self.f(torch.cat([h, e], -1)).squeeze(-1)


def main():
    torch.set_num_threads(os.cpu_count())
    base, ni, head, bank, held_t, LV, ANS = VH.build()
    N = len(held_t); nb = len(bank)

    # ---- replay vhead.py's RNG EXACTLY (same seed, same call order) to recover the states ----
    rng = np.random.default_rng(0)
    tlen = rng.integers(1, VH.MAXT + 1, size=N)
    asked = [rng.choice(nb, size=int(t), replace=False) for t in tlen]
    log(f"states replayed: mean interview length {tlen.mean():.2f} (1..{VH.MAXT})")

    Z = torch.from_numpy(np.load(f"{OUT}/Z.npy"))
    BASE = np.load(f"{OUT}/BASE.npy")
    CU = np.load(f"{OUT}/CU.npy"); CY = np.load(f"{OUT}/CY.npy")
    K = CU.shape[1]
    ADV = torch.from_numpy((CY - CY.mean(1, keepdims=True)).astype(np.float32))
    log(f"targets {N}x{K} | state tail-NDCG {BASE.mean():.4f} | advantage sd {ADV.std():.4f}")

    enc = SetEncoder(ni, token_mode="film", pool="attn")
    ck = torch.load(VH.PB2, map_location="cpu")
    enc.load_state_dict(ck["student"], strict=False); enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    dec = nn.Linear(D, ni); dec.load_state_dict(ck["decoder"])
    Wd = dec.weight.detach(); bd = dec.bias.detach()
    EMB = enc.item_emb.weight.detach()
    Eb = EMB[torch.from_numpy(bank)]                             # (800, D) askable bank embeddings
    W = 1.0 / np.log2(np.arange(2, 12))

    # ---- A2 feature: e^T Sigma_n e, computed BESIDE the recommender ----
    # Sigma_pop = empirical covariance of z across users (the REAL prior: "how do people actually differ")
    Zc = Z - Z.mean(0, keepdim=True)
    Sp = (Zc.T @ Zc) / (N - 1) + 1e-3 * torch.eye(D)
    Spi = torch.linalg.inv(Sp)
    log(f"Sigma_pop: trace {Sp.trace():.3f}  cond ~{torch.linalg.cond(Sp):.1f}")

    def unc_feat(us, cand_idx, lam):
        """e^T Sigma_n e for a batch, with Sigma_n^-1 = Sigma_pop^-1 + lam * sum_t x_t x_t^T (Woodbury, n<=8)."""
        out = torch.zeros(len(us), len(cand_idx[0]) if cand_idx.dim() > 1 else 1)
        for r, u in enumerate(us):
            a = asked[u]; a = a[ANS[u, a]]
            e = Eb[cand_idx[r]]                                  # (K, D)
            if len(a) == 0:
                out[r] = torch.einsum('kd,de,ke->k', e, Sp, e)
                continue
            X = EMB[torch.from_numpy(bank[a])]                   # (n, D)
            X = X / (X.norm(dim=1, keepdim=True) + 1e-8)
            SX = Sp @ X.T                                        # (D, n)
            M = torch.eye(len(a)) / lam.clamp_min(1e-4) + X @ SX # (n, n)
            Sn_e = Sp @ e.T - SX @ torch.linalg.solve(M, X @ (Sp @ e.T))
            out[r] = (e.T * Sn_e).sum(0)
        return out

    # ---- disjoint SELECTION / EVALUATION halves ----
    rs = np.random.default_rng(1); half = rs.random(N) < 0.5
    S = np.where(half)[0]; E = np.where(~half)[0]
    log(f"selection {len(S)}  evaluation {len(E)}  (disjoint, symmetric for every arm and baseline)")

    def tokens(u, extra=None):
        a = asked[u]; a = a[ANS[u, a]]
        if extra is not None and ANS[u, extra]:
            a = np.concatenate([a, [extra]])
        return bank[a], LV[u, a]

    def fold_score(us, picks):
        """Fold each user's PICK and score realized TAIL NDCG@10 through the frozen recommender."""
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
                u = us[i]
                s = sc[r].copy()
                s[bank[asked[u]]] = -1e30                        # everything asked is known
                s[bank[int(picks[i])]] = -1e30
                s[head] = -1e30                                  # TAIL
                hl = set(int(x) for x in held_t[u])
                o = np.argsort(-s)[:10]
                dcg = sum(W[p] for p, t in enumerate(o) if int(t) in hl)
                out[i] = dcg / (W[:min(10, len(hl))].sum() + 1e-12)
        return out

    # ================= TRAIN THE ARMS =================
    lam = nn.Parameter(torch.tensor(1.0))                        # FITTED, never hand-set
    arms = {"A1": QPoint(), "A2": QPoint(nx=2), "A3": QHist()}
    opts = {k: torch.optim.AdamW(list(m.parameters()) + ([lam] if k == "A2" else []), lr=1e-3,
                                 weight_decay=1e-4) for k, m in arms.items()}
    Sn = torch.from_numpy(S)
    for ep in range(EPOCHS):
        perm = Sn[torch.randperm(len(Sn))]
        tot = {k: 0.0 for k in arms}; nbatch = 0
        for b in range(0, len(perm), 256):
            us = perm[b:b + 256].numpy()
            ci = torch.from_numpy(CU[us])                        # (B, K)
            y = ADV[us]
            e = Eb[ci]                                           # (B, K, D)
            z = Z[us].unsqueeze(1).expand(-1, ci.shape[1], -1)
            B, Kc = ci.shape
            # A1
            p = arms["A1"](z.reshape(-1, D), e.reshape(-1, D)).reshape(B, Kc)
            l1 = ((p - y) ** 2).mean()
            opts["A1"].zero_grad(); l1.backward(); opts["A1"].step()
            # A2
            uf = unc_feat(us, ci, lam.detach())
            nt = torch.tensor([[float(ANS[u, asked[u]].sum())] for u in us]).expand(-1, Kc)
            x = torch.stack([uf, nt], -1).reshape(-1, 2)
            p = arms["A2"](z.reshape(-1, D), e.reshape(-1, D), x).reshape(B, Kc)
            l2 = ((p - y) ** 2).mean()
            opts["A2"].zero_grad(); l2.backward(); opts["A2"].step()
            # A3
            batch = [tokens(u) for u in us]
            L = max(1, max(len(t[0]) for t in batch))
            tid = np.zeros((B, L), np.int64); tlv = np.zeros((B, L), np.int64); tp = np.ones((B, L), bool)
            for r, (it, l_) in enumerate(batch):
                k = len(it)
                if k:
                    tid[r, :k] = it; tlv[r, :k] = l_; tp[r, :k] = False
            tok = EMB[torch.from_numpy(tid)].unsqueeze(1).expand(-1, Kc, -1, -1).reshape(B * Kc, L, D)
            tlv_ = torch.from_numpy(tlv).unsqueeze(1).expand(-1, Kc, -1).reshape(B * Kc, L)
            tp_ = torch.from_numpy(tp).unsqueeze(1).expand(-1, Kc, -1).reshape(B * Kc, L)
            p = arms["A3"](tok, tlv_, tp_, e.reshape(-1, D)).reshape(B, Kc)
            l3 = ((p - y) ** 2).mean()
            opts["A3"].zero_grad(); l3.backward(); opts["A3"].step()
            for k, l in zip(arms, [l1, l2, l3]):
                tot[k] += float(l)
            nbatch += 1
            if nbatch % 50 == 0:
                log(f"  ep{ep} b{nbatch}/{len(perm)//256}  " +
                    "  ".join(f"{k} {tot[k]/nbatch:.5f}" for k in arms))
        log(f"[ep{ep+1}] " + "  ".join(f"{k} mse {tot[k]/max(nbatch,1):.5f}" for k in arms))

    # ================= EVALUATE THE POLICY =================
    log("\nEVALUATION: each arm scores ALL 800 candidates, takes its argmax, we fold+score that pick.")
    res = {}
    for name, m in arms.items():
        m.eval(); picks = np.zeros(len(E), np.int64)
        with torch.no_grad():
            for b in range(0, len(E), 512):
                us = E[b:b + 512]
                z = Z[us].unsqueeze(1).expand(-1, nb, -1).reshape(-1, D)
                e = Eb.unsqueeze(0).expand(len(us), -1, -1).reshape(-1, D)
                if name == "A1":
                    q = m(z, e).reshape(len(us), nb)
                elif name == "A2":
                    ci = torch.arange(nb).unsqueeze(0).expand(len(us), -1)
                    uf = unc_feat(us, ci, lam.detach())
                    nt = torch.tensor([[float(ANS[u, asked[u]].sum())] for u in us]).expand(-1, nb)
                    x = torch.stack([uf, nt], -1).reshape(-1, 2)
                    q = m(z, e, x).reshape(len(us), nb)
                else:
                    batch = [tokens(u) for u in us]
                    L = max(1, max(len(t[0]) for t in batch))
                    tid = np.zeros((len(us), L), np.int64); tlv = np.zeros((len(us), L), np.int64)
                    tp = np.ones((len(us), L), bool)
                    for r, (it, l_) in enumerate(batch):
                        k = len(it)
                        if k:
                            tid[r, :k] = it; tlv[r, :k] = l_; tp[r, :k] = False
                    tok = EMB[torch.from_numpy(tid)].unsqueeze(1).expand(-1, nb, -1, -1).reshape(-1, L, D)
                    q = m(tok, torch.from_numpy(tlv).unsqueeze(1).expand(-1, nb, -1).reshape(-1, L),
                          torch.from_numpy(tp).unsqueeze(1).expand(-1, nb, -1).reshape(-1, L),
                          e).reshape(len(us), nb)
                # never re-ask something already asked
                for r, u in enumerate(us):
                    q[r, asked[u]] = -1e30
                picks[b:b + len(us)] = q.argmax(1).numpy()
        res[name] = fold_score(E, picks)
        log(f"  {name}: tail {res[name].mean():.4f}   (distinct picks {len(set(picks.tolist()))}/{nb})")

    # ---- baselines, all with the SAME selection budget ----
    gq = int(np.argmax(ADV[torch.from_numpy(S)].mean(0).numpy() @ np.eye(1) if False else
                       np.bincount(CU[S].ravel(), weights=(CY[S] - CY[S].mean(1, keepdims=True)).ravel(),
                                   minlength=nb) /
                       np.bincount(CU[S].ravel(), minlength=nb).clip(1)))
    res["STATIC"] = fold_score(E, np.full(len(E), gq))
    rr = np.random.default_rng(2)
    res["RANDOM"] = fold_score(E, rr.integers(0, nb, len(E)))
    bo = CU[E][np.arange(len(E)), CY[E].argmax(1)]
    res["BEST-OF-16 (lower bound on the oracle, NOT a ceiling)"] = CY[E].max(1)

    log("\n" + "=" * 76)
    st = res["STATIC"].mean()
    for k in ["RANDOM", "STATIC", "A1", "A2", "A3", "BEST-OF-16 (lower bound on the oracle, NOT a ceiling)"]:
        v = res[k].mean()
        log(f"  {k:<52} tail {v:.4f}   {v-st:+.4f} vs static")
    log("=" * 76)

    rb = np.random.default_rng(3)
    def boot(a, b):
        d = a - b
        bs = np.array([d[rb.integers(0, len(d), len(d))].mean() for _ in range(2000)])
        return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    for k in ["A1", "A2", "A3"]:
        m_, lo, hi = boot(res[k], res["STATIC"])
        log(f"  {k} - STATIC = {m_:+.4f}  95% CI [{lo:+.4f},{hi:+.4f}]  "
            f"{'SIGNIFICANT' if lo > 0 else 'n.s.'}")
    m_, lo, hi = boot(res["A3"], res["A1"])
    log(f"\n  *** A3 - A1 = {m_:+.4f}  95% CI [{lo:+.4f},{hi:+.4f}]   (MDE {MDE}) ***")
    if m_ < MDE:
        log("  => z IS SUFFICIENT. The uncertainty question is CLOSED. No covariance, ever.")
    else:
        m2, _, _ = boot(res["A2"], res["A1"])
        log(f"  => A3 beats A1. A2 recovers {100*m2/max(m_,1e-9):.0f}% of the gap "
            f"({'ship the feature' if m2/max(m_,1e-9) >= 0.7 else 'needs a learned representation = A3'})")
    torch.save({k: m.state_dict() for k, m in arms.items()}, f"{OUT}/arms.pt")
    np.savez(f"{OUT}/results.npz", **{k.split()[0]: v for k, v in res.items()})


if __name__ == "__main__":
    main()
