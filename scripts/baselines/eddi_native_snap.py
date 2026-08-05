r"""eddi_native_snap.py -- NATIVE SNAP of our EDDI implementation against Ma et al., ICML 2019.

Step 1 of the snap ladder (`docs/design/DS_SNAP_LADDER.md`) for the EDDI row. Extends the archived
`scripts/_archive/paper1_v1/eddi_replicate.py` in three ways that the archived script could not
support, all of which the published comparison requires:

  1. THE PAPER'S OWN DATASETS. Ma et al. Figure 3 shows Boston Housing, Energy and Wine (Section 4.2,
     6 UCI datasets, 10% held out as test, 10 repetitions). The archived script used sklearn
     California-housing, which is NOT one of their six, so it could only check the SHAPE of the claim.
     All three Figure-3 datasets are downloaded here from their primary sources.
  2. THE 'SING' ARM. The paper compares three selection strategies: EDDI (per-point adaptive),
     RAND (random order) and SING ("single optimal ordering" -- one fixed global order applied to
     every test point). The archived script had EDDI vs RAND only. SING is the arm that matters to
     us, because it is the tabular analogue of the static global order that currently beats our EDDI
     selector in the Act-1 table. Table 3 (average ranking of AUIC over the 6 datasets, lower is
     better) reports, for the PN parameterisation: EDDI 5.25, Single-best 5.28, Random 7.79.
  3. THE LATENT-ONLY REWARD, i.e. THE ONE WE ACTUALLY SHIP. `src/baselines/eddi_selector.py`
     implements only the FIRST term of the paper's Eq. (9),
         E_{x_i} KL[ q(z|x_i,x_O) || q(z|x_O) ],
     dropping the target-correction term because our cold-start protocol has no designated target
     variable x_phi. The archived replicate implemented a TARGET-oriented reward (expected reduction
     in the predictive variance of x_phi), so it never tested the acquisition our CF selector uses.
     Arm `eddi_latent` below is exactly the shipped reward, on the paper's ruler. If it fails to
     dominate RAND here, the Act-1 EDDI row is measuring a broken acquisition, not a weak idea.

REWARDS IMPLEMENTED (both sample x_i ~ p(x_i|x_O) through the Partial VAE, as the paper prescribes:
z ~ q(z|x_O) then x_i ~ p(x_i|z)):
  eddi_target : gain(i) = Var[x_phi | x_O] - E_{x_i} Var[x_phi | x_i, x_O]
                a Gaussian-predictive proxy for Eq. (6); this is the archived script's reward.
  eddi_latent : gain(i) = E_{x_i} KL[ q(z|x_i,x_O) || q(z|x_O) ]   <-- Eq. (9) first term = OURS.

Usage:
  python scripts/baselines/eddi_native_snap.py --datasets boston,energy,wine,california --seeds 0,1,2
"""
import os
import io
import sys
import time
import json
import zipfile
import argparse
import urllib.request
import numpy as np
import torch
import torch.nn as nn

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".cache", "uci")
CACHE = os.path.abspath(CACHE)
UA = {"User-Agent": "Mozilla/5.0"}


# --------------------------------------------------------------------------- data
def _get(url, name, binary=True):
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, name)
    if not os.path.exists(p):
        d = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120).read()
        open(p, "wb").write(d)
    return open(p, "rb").read() if binary else open(p, "r", errors="replace").read()


def load_dataset(name):
    """-> (A, feature_names, target_name). A is [n, D] with the TARGET as the LAST column."""
    if name == "boston":                       # Ma et al. Fig. 3(a)
        txt = _get("http://lib.stat.cmu.edu/datasets/boston", "boston.txt", binary=False)
        rows = txt.split("\n")[22:]
        vals = []
        for i in range(0, len(rows) - 1, 2):
            a = rows[i].split(); b = rows[i + 1].split()
            if len(a) + len(b) != 14:
                continue
            vals.append([float(v) for v in a + b])
        A = np.asarray(vals, dtype=np.float32)                       # 13 feat + MEDV
        return A, [f"f{i}" for i in range(13)], "MEDV"
    if name == "energy":                       # Ma et al. Fig. 3(b)
        z = zipfile.ZipFile(io.BytesIO(_get(
            "https://archive.ics.uci.edu/static/public/242/energy+efficiency.zip", "energy.zip")))
        import pandas as pd
        df = pd.read_excel(io.BytesIO(z.read([n for n in z.namelist() if n.endswith(".xlsx")][0])))
        df = df.dropna(axis=1, how="all").dropna()
        A = df.values.astype(np.float32)                             # X1..X8, Y1, Y2
        A = np.concatenate([A[:, :8], A[:, 8:9]], axis=1)            # target = Y1 (heating load)
        return A, [f"X{i+1}" for i in range(8)], "Y1"
    if name == "wine":                         # Ma et al. Fig. 3(c)
        z = zipfile.ZipFile(io.BytesIO(_get(
            "https://archive.ics.uci.edu/static/public/186/wine+quality.zip", "wine.zip")))
        raw = z.read("winequality-red.csv").decode()
        lines = [l for l in raw.strip().split("\n")]
        hdr = [h.strip('"') for h in lines[0].split(";")]
        A = np.asarray([[float(v) for v in l.split(";")] for l in lines[1:]], dtype=np.float32)
        return A, hdr[:-1], hdr[-1]
    if name == "california":                   # the archived script's substitute ruler
        from sklearn.datasets import fetch_california_housing
        d = fetch_california_housing()
        A = np.concatenate([d.data.astype(np.float32), d.target.astype(np.float32)[:, None]], 1)
        return A, list(d.feature_names), "MedHouseVal"
    raise SystemExit(f"unknown dataset {name}")


# --------------------------------------------------------------------------- model
class PartialVAE(nn.Module):
    """PN-style point-net set encoder: h([e_j, x_j]) -> mask -> SUM -> head. Same architecture as
    the archived replicate, so the two are comparable."""

    def __init__(self, Dv, zd=10, h=64, kemb=16):
        super().__init__()
        self.Dv, self.zd = Dv, zd
        self.femb = nn.Embedding(Dv, kemb)
        self.enc = nn.Sequential(nn.Linear(kemb + 1, h), nn.ReLU(), nn.Linear(h, h))
        self.head = nn.Linear(h, 2 * zd)
        self.dec = nn.Sequential(nn.Linear(zd, h), nn.ReLU(), nn.Linear(h, Dv))
        self.logsig = nn.Parameter(torch.zeros(Dv))

    def encode(self, x, m):
        B = x.shape[0]
        ids = torch.arange(self.Dv).expand(B, self.Dv)
        tok = torch.cat([self.femb(ids), x.unsqueeze(-1)], -1)
        e = self.enc(tok) * m.unsqueeze(-1)
        mlv = self.head(e.sum(1))
        return mlv[:, :self.zd], mlv[:, self.zd:]

    def decode(self, z):
        return self.dec(z)


def train_pvae(Atr, Dv, updates, seed, log):
    """`updates` = target number of gradient steps, so small (Boston, 506 rows) and large
    (California, 20k rows) datasets get comparable training rather than comparable epochs."""
    torch.manual_seed(seed)
    model = PartialVAE(Dv)
    opt = torch.optim.Adam(model.parameters(), 2e-3)
    steps_per_ep = max(1, int(np.ceil(len(Atr) / 256)))
    epochs = max(10, int(np.ceil(updates / steps_per_ep)))
    for ep in range(epochs):
        perm = torch.randperm(len(Atr))
        for b in range(0, len(perm), 256):
            x = Atr[perm[b:b + 256]]
            m = (torch.rand_like(x) < torch.rand(x.shape[0], 1) * 0.8 + 0.1).float()
            muz, lvz = model.encode(x * m, m)
            z = muz + torch.randn_like(muz) * torch.exp(0.5 * lvz)
            xh = model.decode(z)
            inv = 2 * model.logsig
            nll = (((x - xh) ** 2) / torch.exp(inv) + inv) * m
            kl = -0.5 * (1 + lvz - muz ** 2 - lvz.exp())
            loss = (nll.sum(1) + kl.sum(1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


# --------------------------------------------------------------------------- acquisition
def make_ops(model, Dv, TGT, nz=20):
    def target_pred(x, m):
        with torch.no_grad():
            muz, lvz = model.encode(x * m, m)
            zs = muz.unsqueeze(0) + torch.randn(nz, *muz.shape) * torch.exp(0.5 * lvz)
            xh = model.decode(zs.reshape(-1, model.zd)).reshape(nz, x.shape[0], Dv)[:, :, TGT]
        return xh.mean(0), xh.var(0) + torch.exp(2 * model.logsig[TGT])

    def post(x, m):
        with torch.no_grad():
            return model.encode(x * m, m)
    return target_pred, post


def gauss_kl(m1, lv1, m0, lv0):
    v1, v0 = torch.exp(lv1), torch.exp(lv0)
    return 0.5 * torch.sum(lv0 - lv1 + (v1 + (m1 - m0) ** 2) / v0 - 1.0, dim=-1)


def rewards(model, x, m, feats, TGT, kind, nxi, target_pred, post):
    """[B, Dv] reward matrix (-inf on already-observed / target columns), for one interview state."""
    B, Dv = x.shape
    R = torch.full((B, Dv), -float("inf"))
    with torch.no_grad():
        mu0, lv0 = post(x, m)
        if kind == "eddi_target":
            _, curvar = target_pred(x, m)
        z0 = mu0.unsqueeze(0) + torch.randn(nxi, B, model.zd) * torch.exp(0.5 * lv0)
        xdec = model.decode(z0.reshape(-1, model.zd)).reshape(nxi, B, Dv)   # sampled x_i ~ p(x_i|z)
        for i in feats:
            xi = xdec[:, :, i]                                              # [nxi, B]
            xx = x.unsqueeze(0).repeat(nxi, 1, 1); xx[:, :, i] = xi
            mm = m.unsqueeze(0).repeat(nxi, 1, 1); mm[:, :, i] = 1.0
            xx = xx.reshape(nxi * B, Dv); mm = mm.reshape(nxi * B, Dv)
            if kind == "eddi_target":
                _, nv = target_pred(xx, mm)
                R[:, i] = curvar - nv.reshape(nxi, B).mean(0)
            else:                                                            # eddi_latent (OURS)
                mui, lvi = post(xx, mm)
                kl = gauss_kl(mui, lvi, mu0.repeat(nxi, 1), lv0.repeat(nxi, 1))
                R[:, i] = kl.reshape(nxi, B).mean(0)
    R[:, TGT] = -float("inf")
    R[m.bool()] = -float("inf")
    return R


def run_arm(model, Ate, Dv, TGT, feats, arm, rng, nxi, target_pred, post):
    """-> (rmse_curve [len(feats)+1], per-point abs-error curve [B, len(feats)+1])."""
    B = Ate.shape[0]
    m = torch.zeros(B, Dv); x = Ate * m
    y = Ate[:, TGT]
    curves, errs = [], []
    pm, _ = target_pred(x, m)
    curves.append(torch.sqrt(((pm - y) ** 2).mean()).item()); errs.append((pm - y).abs())
    avail = torch.ones(B, Dv, dtype=torch.bool); avail[:, TGT] = False
    for _step in range(len(feats)):
        if arm == "rand":
            pick = torch.tensor([int(rng.choice(np.flatnonzero(avail[b].numpy()))) for b in range(B)])
        else:
            R = rewards(model, x, m, feats, TGT, arm.replace("sing_", ""), nxi, target_pred, post)
            if arm.startswith("sing_"):
                mean_r = torch.where(torch.isinf(R), torch.full_like(R, float("nan")), R)
                mr = torch.nanmean(mean_r, dim=0)
                mr[torch.isnan(mr)] = -float("inf")   # fully-observed / target columns
                j = int(mr.argmax())                  # ONE global pick, applied to every test point
                pick = torch.full((B,), j, dtype=torch.long)
            else:
                pick = R.argmax(dim=1)
        m[torch.arange(B), pick] = 1.0
        x[torch.arange(B), pick] = Ate[torch.arange(B), pick]
        avail[torch.arange(B), pick] = False
        pm, _ = target_pred(x, m)
        curves.append(torch.sqrt(((pm - y) ** 2).mean()).item()); errs.append((pm - y).abs())
    return curves, torch.stack(errs, dim=1).numpy()


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="boston,energy,wine,california")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--updates", type=int, default=8000,
                    help="gradient steps per Partial-VAE fit (size-normalised across datasets)")
    ap.add_argument("--ntest", type=int, default=150)
    ap.add_argument("--nxi", type=int, default=5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ARMS = ["eddi_target", "eddi_latent", "sing_eddi_target", "sing_eddi_latent", "rand"]
    log = lambda *s: print(*s, flush=True)
    out = {"arms": ARMS, "datasets": {}}

    for ds in a.datasets.split(","):
        A, fnames, tname = load_dataset(ds)
        A = (A - A.mean(0)) / (A.std(0) + 1e-6)
        Dv = A.shape[1]; TGT = Dv - 1; feats = list(range(Dv - 1))
        out["datasets"][ds] = {"n": int(len(A)), "D": int(Dv), "target": tname, "seeds": {}}
        for seed in [int(s) for s in a.seeds.split(",")]:
            rng = np.random.default_rng(seed)
            idx = rng.permutation(len(A))
            ntest = min(a.ntest, max(1, int(0.10 * len(A))))          # paper: 10% held out
            tr, te = idx[ntest:], idx[:ntest]
            Atr, Ate = torch.tensor(A[tr]), torch.tensor(A[te])
            t0 = time.time()
            model = train_pvae(Atr, Dv, a.updates, seed, log)
            target_pred, post = make_ops(model, Dv, TGT)
            res, errmats = {}, {}
            for arm in ARMS:
                c, e = run_arm(model, Ate.clone(), Dv, TGT, feats, arm,
                               np.random.default_rng(seed + 7), a.nxi, target_pred, post)
                res[arm] = c; errmats[arm] = e
                log(f"  [{ds} s{seed}] {arm:18s} RMSE " + " ".join(f"{v:.3f}" for v in c))
            # AUIC per test point (area = mean abs error over the curve), then rank the arms
            auic = np.stack([errmats[arm].mean(axis=1) for arm in ARMS], axis=1)   # [B, n_arms]
            ranks = auic.argsort(axis=1).argsort(axis=1) + 1
            out["datasets"][ds]["seeds"][str(seed)] = {
                "curves": res,
                "auic_mean": {arm: float(auic[:, j].mean()) for j, arm in enumerate(ARMS)},
                "avg_rank": {arm: float(ranks[:, j].mean()) for j, arm in enumerate(ARMS)},
                "minutes": round((time.time() - t0) / 60, 2)}
            log(f"  [{ds} s{seed}] avg AUIC rank (lower=better): " + "  ".join(
                f"{arm}={out['datasets'][ds]['seeds'][str(seed)]['avg_rank'][arm]:.2f}" for arm in ARMS)
                + f"   ({(time.time()-t0)/60:.1f}m)")

    # pooled average ranking, the paper's Table 3 statistic
    log("\n=== POOLED average ranking of AUIC (lower = better) — paper Table 3 analogue ===")
    pooled = {arm: np.mean([s["avg_rank"][arm] for d in out["datasets"].values()
                            for s in d["seeds"].values()]) for arm in ARMS}
    for arm in ARMS:
        log(f"  {arm:18s} {pooled[arm]:.3f}")
    out["pooled_avg_rank"] = {k: float(v) for k, v in pooled.items()}
    if a.out:
        json.dump(out, open(a.out, "w"), indent=2)
        log(f"-> {a.out}")


if __name__ == "__main__":
    main()
