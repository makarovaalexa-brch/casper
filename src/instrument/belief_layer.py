r"""belief_layer.py -- ANALYTIC GAUSSIAN BELIEF (design (ii), DECOUPLED) on the FROZEN i25 tower.

Step-2 design sheet: docs/design/DESIGN_SHEET_STEP2_BELIEF.md (v2.x). Battery spec:
docs/design/GATE_BATTERY_INSTRUMENT.md. Prior art REUSED verbatim where reviewed:
  * the exact Woodbury/dense Sigma algebra + increment objective + pre-fit sign proof from the
    already-adversarially-reviewed scripts/train_precacc.py (the pbC-era PrecAcc). This file ports that
    machinery from the pbC z-space (d=512, halfstar NLEV=15, single-encoder fold) to the i25 tower
    z-space (d=200, NLEV=10, RESIDUAL fold + Arm-A concept latent shift).

DESIGN (ii), the two decoupled halves:
  MEAN  mu(S) = the frozen i25 tower's own nonlinear fold of the ITEM evidence, PLUS the Arm-A concept
        latent shift (post-fold, linear).  Sigma NEVER touches the mean (design (ii) -> G0 bit-identity
        at full profile; empty set -> enc(empty) = the popularity intercept).
          mu(S) = enc_i25(item tokens of S)  +  sum_{c in S} beta(k) * w_c * v_c * d_c
        items move the mean via the certified NONLINEAR tower fold; concepts via a LINEAR latent shift
        along the whitened member direction d_c (Arm A -- the belief-mean update, sheet lines 71-83).
  SIGMA analytic PRECISION ACCUMULATOR, Lambda_t = Lambda0 + sum_j alpha_j phi_j phi_j^T / sigma_j^2 over
        observation directions:
          item     phi_i = RAW decoder row Wd[i]      (F3 repair: raw in the obs equation; the row NORM
                           carries affinity/leverage -- normalization only in leverage read-outs)
          concept  phi_c = whitened member direction  (center + strip top-1 PC of Wd; the G5 recipe)
          refusal  phi   = the probed direction, with a small separately-fitted alpha_refuse (consumption
                           observation, sheet line 86: "no clue about X" = the user does not live there)
        Lambda0 = diag(1/v0), v0 = s0*(var_emp + vfloor)  -- SHRUNK empirical precision from the train-user
        latent population (var_emp = per-dim variance of the tower fold over train users).

THE <=13 FITTED SCALARS (sheet: "per-channel alpha, sigma levels {know-well,vague,refuse}, lambda0"):
  log_alpha  (2 channels {item,concept} x 3 confidence {know-well,vague,refuse}) = 6
  log_s0     global prior-variance scale                                          = 1
  log_vfloor additive variance floor                                              = 1
  TOTAL 8 scalars (<=13).  NOTE on identifiability: the precision contribution is alpha*phi phi^T/sigma^2;
  the product (alpha / sigma^2) is what enters, so the sheet's SEPARATE per-channel alpha and per-level
  sigma^2 are NOT separately identifiable -- we fit their PRODUCT as one positive scalar per
  (channel, confidence) cell (exp-param => PSD => G1a/G1b hold BY CONSTRUCTION). G4's "alpha ordering"
  is read off this product per confidence tier.

SIGMA REPRESENTATION (documented choice): DIAGONAL Lambda0 + LOW-RANK sum of rank-1 atom updates.
  Sigma = (diag(1/v0) + U U^T)^{-1},  U = [sqrt(alpha_j) phi_j].  With d=200 small, BOTH exact routes are
  cheap; we switch on m=|S|: Woodbury (m x m Cholesky) for m <= WOOD_MAX, dense d x d Cholesky otherwise.
  We never materialise a dense 200x200 inverse in the interview regime (m ~ 1..32).

CALIBRATION = held-out-item NLL on val increments, the PrecAcc Huber INCREMENT loss carried forward
  (sheet line 93). Per train user, evidence sub-interview S, held-liked probe i, atom a in S:
    Delta_e = (s*_i - s_i(mu(S\a)))^2 - (s*_i - s_i(mu(S)))^2    [SIGNED; s*_i = fold of the FULL known
              half -> informative atom => Delta_e>0]  (FROZEN, precomputed once, no alpha)
    Delta_v(alpha) = alpha_a (w_i' Sigma(S\a) phi_a)^2 / (1 + alpha_a phi_a' Sigma(S\a) phi_a)   [Sherman-
              Morrison downdate; ONE full-S SigmaOps yields every leave-one-out prediction]
    L = mean Huber_beta(Delta_v - Delta_e).
  PRE-FIT SIGN PROOF (G1 requirement, asserted BEFORE fitting): at alpha_conc=0,
    dL/dalpha_conc < 0  iff  G = sum clamp(Delta_e,-beta,beta) (w_i' Sigma(S\c) phi_c)^2 > 0.  assert G>0.

HARD RULES honoured: score z @ Wd.T + decoder.BIAS (learned bias, NEVER popb); FULL + TAIL always; no LLM;
no retired grids (asserted absent); no data truncation on gates (the alpha-fit's --max_users/--max_atoms
are the same author-SANCTIONED regression subsample as PrecAcc, flagged at the call site).

This module is IMPORT-SAFE and fully exercised by --smoke (synthetic tower, no canonical data touched).
Usage:
  python src/instrument/belief_layer.py --smoke
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")
import sys
import time
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src", "baselines"))
sys.path.insert(0, _HERE)
from train_tower_t2 import level_to_sv, pack_tokens, NLEV, log        # tower forward contract

# retired-answerer ban (Jul-22 audit): must not be reachable from the belief layer
import train_tower_t2 as _T2
assert not hasattr(_T2, "load_answerer"), "retired answerer loader must not exist in the tower module"

D_LAT = 200            # i25 tower latent width (RecVAE latent)
WOOD_MAX = 200         # m <= WOOD_MAX -> Woodbury; else dense d x d Cholesky (both EXACT; cheaper route)
# confidence tiers (sheet: sigma in {know-well, vague, refuse})
KNOW_WELL, VAGUE, REFUSE = 0, 1, 2
CONF_NAMES = ["know-well", "vague", "refuse"]
# channels
CH_ITEM, CH_CONC = 0, 1
CH_NAMES = ["item", "concept"]


# ============================================================================= the <=13 scalars
class BeliefLayer(nn.Module):
    """log_alpha (2 channel x 3 confidence) precision-product cells + log_s0 + log_vfloor = 8 scalars.
    v0 = s0*(var_emp + vfloor) is the SHRUNK anisotropic empirical prior VARIANCE (Lambda0 = diag(1/v0)).
    exp-param => every atom adds PSD precision => tr(Sigma) shrinks and directional shrinkage hold BY
    CONSTRUCTION (G1a/G1b are asserts, not hopes)."""

    def __init__(self, var_emp):
        super().__init__()
        la = torch.zeros(2, 3)                       # alpha init 1.0 for know-well/vague ...
        la[:, REFUSE] = -4.0                         # ... refuse ~0.018 (near-zero info; G4 init ordering)
        self.log_alpha = nn.Parameter(la)
        self.log_s0 = nn.Parameter(torch.zeros(()))                      # global prior-variance scale
        self.log_vfloor = nn.Parameter(torch.full((), math.log(1e-4)))  # additive variance floor
        self.register_buffer("var_emp", torch.as_tensor(var_emp, dtype=torch.float32))

    def alphas(self):
        return torch.exp(self.log_alpha)             # (2,3) > 0  -- the precision PRODUCT per cell

    def v0(self):
        return torch.exp(self.log_s0) * (self.var_emp + torch.exp(self.log_vfloor))   # (d,) > 0

    def conf_precision(self):
        """G4 read-out: effective precision per confidence tier, from the cells the pre-C3 simulation
        populates (items rated -> know-well [item chan]; concept SEL -> vague [concept chan]; refusal
        probe -> refuse). Returns dict tier -> alpha."""
        a = self.alphas().detach().numpy()
        return {"know-well": float(a[CH_ITEM, KNOW_WELL]),
                "vague": float(a[CH_CONC, VAGUE]),
                "refuse": float(a[CH_ITEM, REFUSE])}


# ============================================================================= exact Sigma algebra
class SigmaOps:
    """Exact Sigma = (diag(1/v0) + U U^T)^{-1} ops for one padded batch (ported verbatim from the
    reviewed train_precacc.SigmaOps; d taken from U so it works at d=200). Woodbury for m<=WOOD_MAX,
    dense otherwise. All methods differentiable in U (=> in log_alpha) and v0 (=> s0, vfloor)."""

    def __init__(self, U, v0):
        self.U = U; self.v0 = v0                                          # (B,d,m), (d,)
        self.B, self.d, self.m = U.shape
        self.dense = self.m > WOOD_MAX
        if self.dense:
            Lam = torch.diag(1.0 / v0).unsqueeze(0) + torch.einsum("bdm,bem->bde", U, U)
            self.Lc = torch.linalg.cholesky(Lam)
        else:
            self.Uv = U * v0.view(1, -1, 1)                              # V0 U
            M = torch.eye(self.m).unsqueeze(0) + torch.einsum("bdm,bdk->bmk", U, self.Uv)
            self.Mc = torch.linalg.cholesky(M)                          # zero cols inert -> M stays SPD

    def quad(self, W):
        """w' Sigma w for a SHARED bank W (N,d) -> (B,N)."""
        if self.dense:
            Y = torch.cholesky_solve(W.T.unsqueeze(0).expand(self.B, -1, -1), self.Lc)
            return torch.einsum("nd,bdn->bn", W, Y)
        q0 = (W ** 2) @ self.v0
        T = torch.einsum("nd,bdm->bnm", W, self.Uv)
        sol = torch.cholesky_solve(T.transpose(1, 2), self.Mc)
        return q0.unsqueeze(0) - (T * sol.transpose(1, 2)).sum(-1)

    def quad_user(self, r, W):
        """w' Sigma w for user r with its OWN bank W (n,d) -> (n,)."""
        if self.dense:
            Y = torch.cholesky_solve(W.T, self.Lc[r])
            return torch.einsum("nd,dn->n", W, Y)
        q0 = (W ** 2) @ self.v0
        T = W @ self.Uv[r]
        sol = torch.cholesky_solve(T.T, self.Mc[r])
        return q0 - (T * sol.T).sum(-1)

    def cross_user(self, r, W, Dm):
        """Bilinear W Sigma D^T for user r: W (n,d) probe rows, Dm (mc,d) atom dirs -> (n,mc)."""
        if self.dense:
            Y = torch.cholesky_solve(Dm.T, self.Lc[r])
            return W @ Y
        base = (W * self.v0) @ Dm.T
        TW = W @ self.Uv[r]
        TD = Dm @ self.Uv[r]
        sol = torch.cholesky_solve(TD.T, self.Mc[r])
        return base - TW @ sol

    def trace(self):
        """tr(Sigma) -> (B,)."""
        if self.dense:
            Sig = torch.cholesky_inverse(self.Lc)
            return Sig.diagonal(dim1=1, dim2=2).sum(-1)
        G = torch.einsum("bdm,bdk->bmk", self.Uv, self.Uv)
        sol = torch.cholesky_solve(G, self.Mc)
        return self.v0.sum() - sol.diagonal(dim1=1, dim2=2).sum(-1)


# ============================================================================= observation directions
def build_item_dirs(Wd):
    """ITEM phi_i = RAW decoder row Wd[i] (F3 repair: raw in the observation equation; the row NORM is
    the leverage). Returns (ni, d) float32 -- NOT normalized."""
    return Wd.detach().clone().float()


def build_concept_dirs(Wd, members, tags):
    """CONCEPT phi_c = whitened member centroid (the G5 recipe: center Wd + strip its top-1 PC =
    popularity axis; mean over members; NORMALIZE). Also returns the RAW (un-whitened, popularity-bearing)
    normalized centroid for the G5 floor-blend, and the IDF weight w_c. members: {tag: sid array}."""
    Wd = Wd.detach().float()
    m0 = Wd.mean(0, keepdim=True)
    Ec = Wd - m0
    _, _, Vp = torch.pca_lowrank(Ec, q=1, niter=8)
    pc = F.normalize(Vp[:, 0], dim=0)
    Ew = Ec - torch.outer(Ec @ pc, pc)                                  # popularity-stripped
    d_c = torch.zeros(len(tags), Wd.shape[1]); d_raw = torch.zeros_like(d_c); w_c = torch.zeros(len(tags))
    for i, t in enumerate(tags):
        mem = np.asarray(members[t], np.int64)
        vw = Ew[mem].mean(0); vr = Ec[mem].mean(0)
        d_c[i] = vw / vw.norm().clamp_min(1e-12)
        d_raw[i] = vr / vr.norm().clamp_min(1e-12)
        w_c[i] = 1.0 / math.log1p(len(mem))
    return d_c, d_raw, w_c


# ============================================================================= atoms -> precision U
def build_U(atoms, model, d):
    """atoms: list (per seq) of (dirs (m,d) float tensor, ch (m,) long, conf (m,) long). Returns
    U (B,d,mmax) with column j = sqrt(alpha[ch_j,conf_j]) * dir_j, zero-padded (zero cols inert).
    Differentiable in model.log_alpha through the sqrt(alpha) gather. alpha_override: (2,3) tensor for
    the sign proof (concept row zeroed)."""
    B = len(atoms); mmax = max(1, max(a[0].shape[0] for a in atoms))
    a = model.alphas()
    U = torch.zeros(B, d, mmax)
    cols = []
    for r, (dirs, ch, conf) in enumerate(atoms):
        for j in range(dirs.shape[0]):
            cols.append((r, j, int(ch[j]), int(conf[j]), dirs[j]))
    if cols:
        rr = torch.tensor([c[0] for c in cols]); jj = torch.tensor([c[1] for c in cols])
        chb = torch.tensor([c[2] for c in cols]); cfb = torch.tensor([c[3] for c in cols])
        Dm = torch.stack([c[4] for c in cols])
        Uc = torch.sqrt(a[chb, cfb]).unsqueeze(-1) * Dm
        U = torch.zeros(B, d, mmax, dtype=Uc.dtype)
        U[rr, :, jj] = Uc
    return U


def build_U_override(atoms, alpha, d):
    """Sign-proof helper: same as build_U but with an EXPLICIT (2,3) alpha (concept row zeroed)."""
    B = len(atoms); mmax = max(1, max(a[0].shape[0] for a in atoms))
    U = torch.zeros(B, d, mmax)
    cols = []
    for r, (dirs, ch, conf) in enumerate(atoms):
        for j in range(dirs.shape[0]):
            cols.append((r, j, int(ch[j]), int(conf[j]), dirs[j]))
    if cols:
        rr = torch.tensor([c[0] for c in cols]); jj = torch.tensor([c[1] for c in cols])
        chb = torch.tensor([c[2] for c in cols]); cfb = torch.tensor([c[3] for c in cols])
        Dm = torch.stack([c[4] for c in cols])
        Uc = torch.sqrt(alpha[chb, cfb].clamp_min(0.0)).unsqueeze(-1) * Dm
        U = torch.zeros(B, d, mmax, dtype=Uc.dtype)
        U[rr, :, jj] = Uc
    return U


# ============================================================================= the MEAN (tower + Arm A)
def fold_items(enc, item_seqs, batch=256):
    """Frozen i25 tower fold of ITEM evidence only. item_seqs: list of (sids np, levels np). Returns
    z (B,d) float32. Empty seqs -> enc(empty) = the popularity intercept (gate g(0)=0)."""
    B = len(item_seqs)
    Z = torch.zeros(B, enc.d_out)
    enc.eval()
    with torch.no_grad():
        for st in range(0, B, batch):
            chunk = item_seqs[st:st + batch]
            rows = [(np.asarray(s, np.int64), np.asarray(l, np.int64), level_to_sv(np.asarray(l)))
                    for s, l in chunk]
            ids, vals, pad, lvs = pack_tokens(rows)
            Z[st:st + len(chunk)] = enc(ids, vals, pad, lvs)
    return Z


class ConceptMean:
    """Arm-A concept latent shift operator (the sheet's belief-mean update for concepts):
        shift(c, v) = beta_ctx * w_c * v * d_c
    with the G5-fix COLD override at k=0 (per-k beta and/or the raw-centroid floor blend). v = valence
    sign from the concept answer level. Holds d_c (whitened), d_raw (popularity-bearing), w_c."""

    def __init__(self, d_c, d_raw, w_c, beta_ctx=5.0, beta_cold=None, floor_rho=0.0):
        self.d_c = d_c; self.d_raw = d_raw; self.w_c = w_c
        self.beta_ctx = beta_ctx
        self.beta_cold = beta_ctx if beta_cold is None else beta_cold
        self.floor_rho = floor_rho                     # G5 floor-blend: raw-centroid fraction at k=0 only

    def dir_cold(self, c):
        """Cold (k=0) mean direction: blend a fraction of the popularity-bearing raw centroid back in."""
        if self.floor_rho <= 0:
            return self.d_c[c]
        mix = (1.0 - self.floor_rho) * self.d_c[c] + self.floor_rho * self.d_raw[c]
        return mix / mix.norm().clamp_min(1e-12)

    def shift(self, c, v, cold=False):
        beta = self.beta_cold if cold else self.beta_ctx
        d = self.dir_cold(int(c)) if cold else self.d_c[int(c)]
        return beta * float(self.w_c[int(c)]) * float(v) * d


def mean_fold(enc, item_seqs, conc_lists, cmean, cold_ctx=False):
    """FULL decoupled mean per seq: tower fold of items + sum of Arm-A concept shifts.
    conc_lists: list (per seq) of [(cid, valence), ...]. cold_ctx: use the G5 cold operator (k=0)."""
    Z = fold_items(enc, item_seqs)
    for r, cl in enumerate(conc_lists):
        for cid, v in cl:
            Z[r] = Z[r] + cmean.shift(cid, v, cold=cold_ctx)
    return Z


# ============================================================================= empirical prior variance
def empirical_var(enc, item_seqs, batch=256):
    """var_emp = per-dim variance of the tower fold mu over a train-user population (the shrunk empirical
    precision source, sheet: Lambda0 from the train-user latent population). item_seqs = full known-half
    item tokens per train user."""
    Z = fold_items(enc, item_seqs, batch=batch).numpy()
    v = Z.var(axis=0)
    return np.maximum(v, 1e-6).astype(np.float32)


# ============================================================================= increment calibration
def _atoms_for_seq(item_ids, item_lv, conc, cmean, Wd, item_dirs):
    """Assemble the Sigma atoms (dirs, ch, conf) for one evidence seq S:
      items -> RAW Wd[i] dir, channel item, confidence know-well
      concepts (answered) -> whitened d_c, channel concept, confidence vague
      concepts (refusal, valence 0) -> whitened d_c, channel item(probe)->refuse OR concept refuse
    Returns (dirs (m,d), ch (m,), conf (m,)) plus a parallel list of ('item'|'conc', key, valence) for
    the leave-one-out mean bookkeeping."""
    dirs = []; ch = []; conf = []; meta = []
    for sid, lv in zip(item_ids, item_lv):
        dirs.append(item_dirs[int(sid)]); ch.append(CH_ITEM); conf.append(KNOW_WELL)
        meta.append(("item", int(sid), int(lv)))
    for cid, v in conc:
        dirs.append(cmean.d_c[int(cid)])
        if v == 0:                                    # refusal = consumption observation (small alpha)
            ch.append(CH_CONC); conf.append(REFUSE)
        else:
            ch.append(CH_CONC); conf.append(VAGUE)
        meta.append(("conc", int(cid), int(v)))
    if dirs:
        return (torch.stack(dirs), np.asarray(ch, np.int64), np.asarray(conf, np.int64)), meta
    return (torch.zeros(0, Wd.shape[1]), np.zeros(0, np.int64), np.zeros(0, np.int64)), meta


def precompute_increments(enc, Wd, bd, item_dirs, cmean, users, rng_seed, max_atoms=0, drop=0.6):
    """Delta_e precompute (frozen, no alpha, ONCE). Per train user u = {items,levels,liked}: build an
    evidence sub-interview S (heavy item dropout + the user's SEL concepts), held-liked probes tg (never
    in S), TARGET s*_i = s_i(mu(FULL known half)); for each atom a in S the leave-one-out fold mu(S\a) ->
        De[i,a] = (s*_i - s_i(mu(S\a)))^2 - (s*_i - s_i(mu(S)))^2   [SIGNED]
    Returns a list aligned with `users` (None where unusable). Each entry: dirs, ch, conf, tg, De."""
    out = []
    for ui, u in enumerate(users):
        rng = np.random.default_rng(rng_seed + ui)
        items = np.asarray(u["items"], np.int64); lv = np.asarray(u["levels"], np.int64)
        liked = np.asarray(u["liked"], np.int64)
        if len(items) < 3 or len(liked) < 2:
            out.append(None); continue
        # probes = a held slice of liked; evidence drawn from the REST (leak-free)
        nprobe = max(1, len(liked) // 3)
        tg = rng.choice(liked, size=min(nprobe, len(liked)), replace=False).astype(np.int64)
        keep = ~np.isin(items, tg)
        ev_items = items[keep]; ev_lv = lv[keep]
        if len(ev_items) == 0:
            out.append(None); continue
        # sub-interview item dropout
        m = rng.random(len(ev_items)) >= drop
        if not m.any():
            m[rng.integers(0, len(ev_items))] = True
        S_items = ev_items[m]; S_lv = ev_lv[m]
        conc = list(u.get("concepts", []))            # [(cid, valence)]
        atoms, meta = _atoms_for_seq(S_items, S_lv, conc, cmean, Wd, item_dirs)
        na = atoms[0].shape[0]
        if na == 0:
            out.append(None); continue
        Wt = Wd[torch.from_numpy(tg)]; bt = bd[torch.from_numpy(tg)]
        # full known-half fold = ALL items at real levels + the concepts -> target destination
        mu_full = mean_fold(enc, [(items, lv)], [conc], cmean)[0]
        mu_S = mean_fold(enc, [(S_items, S_lv)], [conc], cmean)[0]
        s_star = mu_full @ Wt.T + bt
        s_S = mu_S @ Wt.T + bt
        term2 = (s_star - s_S) ** 2
        # atom subset (SANCTIONED regression cap; folds still use the full S)
        idx = np.arange(na)
        if max_atoms and na > max_atoms:
            idx = np.sort(rng.choice(na, max_atoms, replace=False))
        De = np.empty((len(tg), len(idx)), np.float32)
        # leave-one-out mean folds
        loo_item_seqs = []; loo_conc = []; loo_key = []
        for a in idx:
            kind, key, val = meta[a]
            if kind == "item":
                sel = [j for j in range(len(S_items)) if not (S_items[j] == key)]
                # remove exactly one occurrence at atom position among items
                it_positions = [p for p, mt in enumerate(meta[:len(S_items)])]
                loo_item_seqs.append((np.delete(S_items, a), np.delete(S_lv, a)))
                loo_conc.append(conc)
            else:
                cpos = a - len(S_items)
                loo_item_seqs.append((S_items, S_lv))
                loo_conc.append([cc for k, cc in enumerate(conc) if k != cpos])
            loo_key.append(a)
        loo_mu = mean_fold(enc, loo_item_seqs, loo_conc, cmean)
        for jj, a in enumerate(idx):
            s_loo = loo_mu[jj] @ Wt.T + bt
            De[:, jj] = (((s_star - s_loo) ** 2) - term2).numpy()
        dirs, ch, conf = atoms
        out.append(dict(dirs=dirs[idx], ch=ch[idx], conf=conf[idx], tg=tg, De=De))
    return out


def prefit_sign_proof(model, Wd, inc, d, huber_beta=1.0):
    r"""MANDATORY G1 gate (seconds, BEFORE the fit). At alpha_conc=0:
        G = sum_{concept triples} clamp(De,-beta,beta) * (w_i' Sigma(S\c) phi_c)^2 > 0  <=>  dL/dalpha_c<0.
    Concept alpha row zeroed EXACTLY (sqrt(0)=0 -> inert cols -> Sigma(S)==Sigma(S\c)). assert G>0."""
    a0 = model.alphas().detach().clone(); a0[CH_CONC, :] = 0.0
    ex = [e for e in inc if e is not None and (e["ch"] == CH_CONC).any()]
    assert ex, "sign proof: no evidence seq contains a concept atom -- cannot certify (data problem)"
    v0 = model.v0().detach()
    Gtot = 0.0; ntrip = 0
    with torch.no_grad():
        for st in range(0, len(ex), 128):
            ce = ex[st:st + 128]
            atoms = [(e["dirs"], e["ch"], e["conf"]) for e in ce]
            U = build_U_override(atoms, a0, d)
            ops = SigmaOps(U, v0)
            for r, e in enumerate(ce):
                cc = np.flatnonzero(e["ch"] == CH_CONC)
                Dm = e["dirs"][cc]
                P = ops.cross_user(r, Wd[torch.from_numpy(e["tg"])], Dm)      # (n, mc)
                Dcl = torch.from_numpy(e["De"][:, cc]).clamp(-huber_beta, huber_beta)
                Gtot += float((Dcl * P ** 2).sum())
                ntrip += e["De"].shape[0] * len(cc)
    log(f"[sign-proof] G={Gtot:.6g} over {ntrip:,} concept triples (Huber beta={huber_beta})")
    assert Gtot > 0, (f"SIGN PROOF FAILED: G={Gtot:.6g} <= 0 -- alpha_conc would stay at zero "
                      "(dead concept dirs / all-zero De). STOP, do not train.")
    return {"G": float(Gtot), "n_triples": int(ntrip), "huber_beta": float(huber_beta)}


def fit_alphas(model, Wd, inc, d, epochs=2, lr=0.05, huber=1.0):
    """INCREMENT-objective fit of the <=13 scalars (touches NO encoder -- Delta_e is frozen). Per user
    one differentiable SigmaOps on the full S; the exact downdate identity predicts each LOO reduction:
        Delta_v_a = alpha_a (w' Sigma(S) phi_a)^2 / (1 - alpha_a phi_a' Sigma(S) phi_a).
    L = mean Huber_beta(Delta_v - Delta_e)."""
    ex = [e for e in inc if e is not None]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    order = np.argsort([e["dirs"].shape[0] for e in ex])
    batches = [order[i:i + 128] for i in range(0, len(order), 128)]
    hist = []
    for ep in range(epochs):
        rng = np.random.default_rng(500 + ep); bs = list(batches); rng.shuffle(bs)
        run = 0.0; nb = 0; nclamp = 0; natom = 0
        for bat in bs:
            be = [ex[i] for i in bat]
            atoms = [(e["dirs"], e["ch"], e["conf"]) for e in be]
            opt.zero_grad()
            U = build_U(atoms, model, d)
            ops = SigmaOps(U, model.v0())
            a = model.alphas()
            closses = []
            for r, e in enumerate(be):
                Dm = e["dirs"]
                alp = a[torch.from_numpy(e["ch"]), torch.from_numpy(e["conf"])]     # (m,)
                q = ops.quad_user(r, Dm)
                P = ops.cross_user(r, Wd[torch.from_numpy(e["tg"])], Dm)
                rden = 1.0 - alp * q
                nclamp += int((rden.detach() < 1e-6).sum()); natom += len(alp)
                dv = alp.unsqueeze(0) * P ** 2 / rden.clamp_min(1e-6).unsqueeze(0)
                De = torch.from_numpy(e["De"])
                closses.append(F.smooth_l1_loss(dv, De, beta=huber, reduction="mean"))
            loss = torch.stack(closses).mean()
            loss.backward(); opt.step()
            run += float(loss); nb += 1
        hist.append(dict(ep=ep, loss=run / max(nb, 1), clamp_frac=nclamp / max(natom, 1)))
        aa = model.alphas().detach().numpy()
        log(f"[fit ep{ep+1}] L={run/max(nb,1):.5f} clamp={nclamp}/{natom} "
            f"a_item={np.round(aa[0],3).tolist()} a_conc={np.round(aa[1],3).tolist()} "
            f"s0={float(torch.exp(model.log_s0)):.3f}")
    return hist


# ============================================================================= G5 FIX (author priority)
def g5_fix_perk_beta(concept_score_fn, betas=(0.0, 0.5, 1.0, 2.0, 5.0)):
    """G5-FIX option 1 -- PER-K beta. concept_score_fn(beta, cold) -> mean concept-only NDCG@10 on val.
    At k=0 (cold) beta is calibrated SEPARATELY from the k>=2 (context) beta. Returns the per-k beta
    that maximises the concept-only NDCG at that k (subject to clearing the intercept at k=0). This is a
    thin harness -- the real val sweep runs inside run_battery_phaseB (build+smoke here)."""
    cold = {b: concept_score_fn(b, cold=True) for b in betas}
    ctx = {b: concept_score_fn(b, cold=False) for b in betas}
    best_cold = max(cold, key=lambda b: cold[b]); best_ctx = max(ctx, key=lambda b: ctx[b])
    return {"cold_curve": cold, "ctx_curve": ctx, "beta_cold": best_cold, "beta_ctx": best_ctx}


def g5_fix_floor_blend(concept_score_fn, rhos=(0.0, 0.25, 0.5, 0.75, 1.0), beta_cold=2.0):
    """G5-FIX option 2 -- FLOOR BLEND. Retain a fraction rho of the popularity-bearing RAW member
    centroid in the COLD (k=0) mean direction only. concept_score_fn(rho) -> mean concept-only k=0
    NDCG@10 on val. Returns the rho that maximises it (clearing the intercept). k>=2 uses the pure
    whitened direction (rho=0) so the +0.0084 context result is untouched."""
    curve = {r: concept_score_fn(r) for r in rhos}
    best = max(curve, key=lambda r: curve[r])
    return {"cold_curve": curve, "floor_rho": best, "beta_cold": beta_cold}


# ============================================================================= SMOKE
def _smoke():
    import recvae as R
    from train_tower_t2 import build_model
    print("[SMOKE] synthetic i25 tower + belief layer (NOT the canonical split)")
    rng = np.random.RandomState(0)
    ni, nu, d = 130, 80, 16
    cnt = rng.rand(ni) * 100
    fake = R.RecVAE(24, d, ni)
    a = argparse.Namespace(arch="i25", teacher="warm_init", t_hidden=24, t_latent=d, token="film",
                           train_decoder=False, sign_prior=True, unfreeze_emb=False, lr=3e-4,
                           warm_lr_scale=0.1, full_kd=False, full_kd_w=0.3)
    enc, decoder, teacher, params, groups = build_model(a, ni, cnt, teacher_override=fake)
    with torch.no_grad():                                    # nonzero rho so folds move z
        for p in enc.rho[-1].parameters():
            p.add_(torch.randn_like(p) * 0.05)
    enc.eval()
    Wd = decoder.weight.detach(); bd = decoder.bias.detach()
    item_dirs = build_item_dirs(Wd)
    assert item_dirs.shape == (ni, d)
    # concepts: random membership
    tags = list(range(6))
    members = {t: np.sort(rng.choice(ni, size=rng.randint(20, 40), replace=False)) for t in tags}
    d_c, d_raw, w_c = build_concept_dirs(Wd, members, tags)
    assert d_c.shape == (6, d) and torch.allclose(d_c.norm(dim=1), torch.ones(6), atol=1e-4)
    cmean = ConceptMean(d_c, d_raw, w_c, beta_ctx=5.0)
    print(f"[SMOKE] dirs OK: item raw (norm range {item_dirs.norm(dim=1).min():.2f}"
          f"..{item_dirs.norm(dim=1).max():.2f}); concept whitened unit")

    # synthetic train users (items + a couple SEL concepts each)
    users = []
    for _ in range(nu):
        k = rng.randint(8, 30)
        its = rng.choice(ni, size=k, replace=False).astype(np.int64)
        lvs = rng.randint(0, NLEV, size=k).astype(np.int64)
        lvs[:3] = [8, 9, 8]
        liked = its[lvs >= 7]
        if len(liked) < 2:
            liked = its[:2]; lvs[:2] = 8
        conc = [(int(rng.choice(tags)), 1)]
        users.append({"items": its, "levels": lvs, "liked": liked.astype(np.int64), "concepts": conc})

    # empirical prior variance -> belief layer
    var_emp = empirical_var(enc, [(u["items"], u["levels"]) for u in users])
    assert var_emp.shape == (d,)
    model = BeliefLayer(var_emp)
    npar = sum(p.numel() for p in model.parameters())
    assert npar == 8, f"expected 8 scalars, got {npar}"
    print(f"[SMOKE] BeliefLayer PASS: {npar} scalars (log_alpha 2x3 + s0 + vfloor); "
          f"v0 mean {float(model.v0().mean()):.4g}")

    # mean decoupling: empty set -> intercept (bit-identity to bias), full set != empty
    z_empty = fold_items(enc, [(np.zeros(0, np.int64), np.zeros(0, np.int64))])[0]
    sc_empty = z_empty @ Wd.T + bd
    assert torch.allclose(sc_empty, bd, atol=1e-5), "empty-set mean must decode to the bias (intercept)"
    print("[SMOKE] mean decoupling PASS: enc(empty) decodes to the decoder bias (intercept)")

    # Sigma by construction: tr(Sigma) shrinks as atoms are added; directional along phi_c
    atoms_a = [(item_dirs[users[0]["items"][:2]], np.array([CH_ITEM, CH_ITEM]), np.array([KNOW_WELL]*2))]
    atoms_b = [(item_dirs[users[0]["items"][:6]], np.array([CH_ITEM]*6), np.array([KNOW_WELL]*6))]
    tr_a = float(SigmaOps(build_U(atoms_a, model, d), model.v0()).trace()[0])
    tr_b = float(SigmaOps(build_U(atoms_b, model, d), model.v0()).trace()[0])
    tr_0 = float(model.v0().sum())
    assert tr_0 > tr_a > tr_b, f"tr(Sigma) must shrink: {tr_0:.3f} {tr_a:.3f} {tr_b:.3f}"
    print(f"[SMOKE] G1a-by-construction PASS: tr(Sigma) k0={tr_0:.2f} > k2={tr_a:.2f} > k6={tr_b:.2f}")
    # directional: fold concept c -> variance along d_c drops much more than a random other dir
    ops0 = SigmaOps(build_U([(torch.zeros(0, d), np.zeros(0, np.int64), np.zeros(0, np.int64))], model, d),
                    model.v0())
    c0 = 0
    ops1 = SigmaOps(build_U([(d_c[c0:c0+1], np.array([CH_CONC]), np.array([VAGUE]))], model, d), model.v0())
    q0 = float(ops0.quad(d_c[c0:c0+1])[0, 0]); q1 = float(ops1.quad(d_c[c0:c0+1])[0, 0])
    other = float(ops1.quad(d_c[1:2])[0, 0]); q0o = float(ops0.quad(d_c[1:2])[0, 0])
    own_drop = (q0 - q1) / q0; oth_drop = (q0o - other) / q0o
    assert own_drop > 2 * max(oth_drop, 1e-9), f"directional shrinkage own {own_drop:.3f} not >=2x other {oth_drop:.3f}"
    print(f"[SMOKE] G1b-by-construction PASS: own-dir drop {own_drop:.3f} >= 2x other {oth_drop:.3f}")

    # increments -> sign proof -> fit
    inc = precompute_increments(enc, Wd, bd, item_dirs, cmean, users, rng_seed=910_000, max_atoms=12)
    nusable = sum(1 for e in inc if e is not None)
    allDe = np.concatenate([e["De"].ravel() for e in inc if e is not None])
    print(f"[SMOKE] increments PASS: {nusable}/{nu} usable; De {float((allDe>0).mean()):.0%} pos "
          f"(signed target working)")
    sign = prefit_sign_proof(model, Wd, inc, d, huber_beta=1.0)
    print(f"[SMOKE] sign proof PASS: G={sign['G']:.4g} > 0 (alpha_conc provably leaves zero)")
    a_before = model.alphas().detach().clone()
    hist = fit_alphas(model, Wd, inc, d, epochs=2, lr=0.05)
    assert not torch.equal(model.alphas().detach(), a_before), "fit did not move alpha"
    cp = model.conf_precision()
    print(f"[SMOKE] fit PASS: final loss {hist[-1]['loss']:.5f}; conf precision {cp}")

    # G5-fix harnesses (synthetic scorer: cold prefers popularity-blend, ctx prefers pure whitened)
    def fake_concept_score(beta, cold):
        base = 0.128
        return base + (0.001 * beta if not cold else -0.002 * beta)     # ctx up, cold down w/ beta
    perk = g5_fix_perk_beta(fake_concept_score)
    assert perk["beta_cold"] == 0.0, "per-k beta: cold should prefer the smallest beta here"
    def fake_floor_score(rho):
        return 0.126 + 0.01 * rho                                        # more popularity -> clears intercept
    fb = g5_fix_floor_blend(fake_floor_score)
    assert fb["floor_rho"] == 1.0
    print(f"[SMOKE] G5-fix harnesses PASS: per-k beta_cold={perk['beta_cold']} beta_ctx={perk['beta_ctx']}; "
          f"floor best rho={fb['floor_rho']}")
    # ConceptMean cold operator actually blends in raw centroid
    cm2 = ConceptMean(d_c, d_raw, w_c, beta_ctx=5.0, beta_cold=2.0, floor_rho=0.5)
    s_ctx = cm2.shift(0, 1, cold=False); s_cold = cm2.shift(0, 1, cold=True)
    assert not torch.allclose(s_ctx / s_ctx.norm(), s_cold / s_cold.norm(), atol=1e-4), \
        "cold floor-blend must change the shift direction"
    print("[SMOKE] ConceptMean floor-blend PASS: cold direction != context direction")
    print("[SMOKE] COMPLETE (belief_layer: dirs, Sigma algebra, mean decoupling, increments, "
          "sign proof, fit, G5-fix)")


# ============================================================================= REAL FIT (author launches)
def _build_train_fit_data(snapshot, max_users, min_members):
    """Assemble train-user calibration data on the canonical split (CPU-heavy -- author launches):
    graded profiles (items/levels/liked) + a SEL top concept per user, the frozen i25 tower, item dirs,
    whitened concept dirs, ConceptMean, and var_emp. Returns (enc, Wd, bd, item_dirs, cmean, users)."""
    import argparse as _ap
    import pandas as pd
    from train_tower_t2 import (reproduce_partition, build_train_profiles, build_model, compute_head_mask)
    PROC = os.path.join(_ROOT, "data", "ml-25m", "proc")
    GENOME = os.path.join(_ROOT, "data", "movielens", "genome-scores.csv")
    import metrics as _M
    meta = _M.load_meta(PROC); ni = meta["n_items"]
    train_mat = _M.load_train(ni, PROC)
    _, cnt = compute_head_mask(train_mat, ni)
    unique_uid, tr_set, vd_set, te_set, n_train, raw, show2id, usid = reproduce_partition()
    users = build_train_profiles(raw, tr_set, show2id, max_users=(max_users or None))
    a = _ap.Namespace(arch="i25", teacher="warm_init", t_hidden=600, t_latent=200, token="film",
                      train_decoder=False, sign_prior=True, unfreeze_emb=False, lr=3e-4,
                      warm_lr_scale=0.1, full_kd=False, full_kd_w=0.3)
    enc, decoder, teacher, params, groups = build_model(a, ni, cnt)
    blob = torch.load(snapshot, map_location="cpu")
    enc.load_state_dict(blob["enc"]); decoder.load_state_dict(blob["decoder"]); enc.eval()
    Wd = decoder.weight.detach().float(); bd = decoder.bias.detach().float()
    # genome concepts
    m2s = {int(m): i for i, m in enumerate(usid)}
    g = pd.read_csv(GENOME); g = g[g["relevance"] >= 0.5]; g = g[g["movieId"].isin(m2s)]
    g["sid"] = g["movieId"].map(m2s).astype(np.int64)
    members = {int(t): np.sort(sub["sid"].values) for t, sub in g.groupby("tagId")}
    members = {t: mm for t, mm in members.items() if len(mm) >= min_members}
    tags = sorted(members.keys())
    d_c, d_raw, w_c = build_concept_dirs(Wd, members, tags)
    cmean = ConceptMean(d_c, d_raw, w_c, beta_ctx=5.0)
    # SEL top concept per train user (member like-rate lift; held targets N/A for train -- full profile)
    grate = np.array([cnt[members[t]].sum() for t in tags]) / max(cnt.sum(), 1e-9)
    for u in users:
        liked = np.asarray(u["liked"], np.int64)
        if len(liked) == 0:
            u["concepts"] = []; continue
        hit = np.array([np.isin(members[t], liked).sum() for t in tags], np.float64)
        lift = (hit / max(len(liked), 1)) / np.maximum(grate, 1e-12)
        lift[hit < 2] = -np.inf
        u["concepts"] = [(int(np.argmax(lift)), 1)] if np.isfinite(lift.max()) else []
    item_dirs = build_item_dirs(Wd)
    return enc, Wd, bd, item_dirs, cmean, users


def fit_main(args):
    snap = args.snapshot or os.path.join(_ROOT, ".cache", "instrument", "t2i25_EP4_SNAP.pt")
    log(f"[fit] REAL calibration on {os.path.basename(snap)} "
        f"(max_users={args.max_users or 'all'} max_atoms={args.max_atoms or 'all'} -- "
        "author-SANCTIONED regression subsample if set; HARD RULE #1)")
    enc, Wd, bd, item_dirs, cmean, users = _build_train_fit_data(snap, args.max_users, args.min_members)
    var_emp = empirical_var(enc, [(np.asarray(u["items"], np.int64),
                                   np.asarray(u["levels"], np.int64)) for u in users])
    model = BeliefLayer(var_emp)
    d = enc.d_out
    inc = precompute_increments(enc, Wd, bd, item_dirs, cmean, users, rng_seed=910_000,
                                max_atoms=args.max_atoms)
    sign = prefit_sign_proof(model, Wd, inc, d, huber_beta=args.huber)      # MANDATORY: assert G>0
    hist = fit_alphas(model, Wd, inc, d, epochs=args.epochs, lr=args.lr, huber=args.huber)
    ckpt = args.out or os.path.join(_ROOT, ".cache", "instrument", "belief_i25.pt")
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    torch.save({"belief": model.state_dict(), "sign_proof": sign, "fit_hist": hist,
                "conf_precision": model.conf_precision(), "snapshot": os.path.basename(snap),
                "var_emp_mean": float(var_emp.mean())}, ckpt)
    log(f"[fit] DONE -> {ckpt} | conf precision {model.conf_precision()} | G={sign['G']:.4g}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--fit", action="store_true", help="REAL increment calibration (author launches)")
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--huber", type=float, default=1.0)
    ap.add_argument("--min_members", type=int, default=30)
    ap.add_argument("--max_users", type=int, default=0, help="SANCTIONED fit subsample (0=all)")
    ap.add_argument("--max_atoms", type=int, default=0, help="SANCTIONED LOO-atom cap/user (0=all)")
    args = ap.parse_args()
    if args.smoke:
        _smoke()
    elif args.fit:
        fit_main(args)
    else:
        ap.error("belief_layer.py is a library; --smoke for the code-path check, --fit for real calibration")
