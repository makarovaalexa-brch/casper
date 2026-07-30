r"""i26_encoder.py -- the interview-native arm: an EXPOSURE branch + a trainable per-item prior.

SPEC: docs/design/RETRAIN_DESIGN_SHEET.md section 4. This is a NEW ARM (`--arch i26`). The certified
i25 path in train_tower_t2.py is untouched, so `t2final_best.pt` stays exactly reproducible.

    z     = native_z*a(n) + g(n) * rho_taste([sum_phi, a*native_z, log1p(n), a])   [i25, UNCHANGED]
            + g_e(m) * rho_expo([sum_psi, log1p(m)])                               [NEW]
    score = z @ Wd' + bd + delta_b                                                 [NEW]

WHY AN EXPOSURE BRANCH AND NOT AN 11th LEVEL. An 11th gamma band would push "asked but not seen" through
the TASTE pathway as a scalar -- i.e. encode exposure as a weak dislike. Exposure is not taste: most
unseen films are unseen for reasons unrelated to preference, and under HELF there are ~4 unseen tokens
per answered one, so that noise would swamp the signal. A separate branch keeps the two distinct.

WHY delta_b. Measured 2026-07-30: the population marginal is NOT in the frozen decoder's span. Ridge-
fitting a latent to reproduce log-popularity gives Spearman 0.836 but NDCG@10 = 0.0032 -- a 200-dim
linear decoder tracks popularity loosely across 18,359 items and gets the TOP TEN wrong, and NDCG only
sees the top ten. So no amount of encoder training can make our zero-evidence prediction equal counting.
A trainable per-item bias is the minimal principled fix; the RecVAE decoder itself stays frozen.

INITIALISATION IS THE SAFETY PROPERTY. `rho_expo`'s last layer and `delta_b` are BOTH zero-init, so at
step 0 this arm is BIT-IDENTICAL to i25. Every change has to be earned by the loss. `test_i26_identity`
asserts exactly that.

UNSEEN TOKENS ride the existing (ids, vals, pad, lvs) signature with the sentinel level UNSEEN_LEVEL =
NLEV. They are routed AWAY from gamma/beta/phi and into psi/rho_expo, and they are excluded from the
frozen native anchor and from the taste token count n. No new input tensor, no changed call signature.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NLEV = 10
UNSEEN_LEVEL = NLEV          # sentinel: "asked, and the user has not seen it"
LIKE_MIN_LEVEL = 7


class I26Encoder(nn.Module):
    """i25 plus an exposure branch. Bit-identical to i25 at init (zero-init rho_expo output)."""

    def __init__(self, ni, native_encoder, d_lat=200, h=512, token_mode="film", n_concepts=0,
                 h_expo=256):
        super().__init__()
        self.ni = ni; self.d = d_lat; self.d_out = d_lat; self.token_mode = token_mode
        self.norm_feat = False
        self._native = [native_encoder]                       # UNREGISTERED (frozen, external)
        self.item_emb = nn.Embedding(ni, d_lat)
        self.nc = n_concepts
        if n_concepts > 0:
            self.concept_emb = nn.Embedding(n_concepts, d_lat)
            nn.init.normal_(self.concept_emb.weight, std=0.02)
        # ---- taste path (i25, unchanged) -------------------------------------------------
        self.gamma = nn.Embedding(NLEV, d_lat); self.beta = nn.Embedding(NLEV, d_lat)
        nn.init.ones_(self.gamma.weight); nn.init.normal_(self.beta.weight, std=0.02)
        self.phi = nn.Sequential(nn.Linear(d_lat, h), nn.GELU(), nn.Linear(h, h))
        self.rho = nn.Sequential(nn.Linear(h + d_lat + 2, h), nn.GELU(), nn.Linear(h, d_lat))
        nn.init.zeros_(self.rho[-1].weight); nn.init.zeros_(self.rho[-1].bias)
        self.gate_a = nn.Parameter(torch.tensor(0.5413))
        self.anchor_b = nn.Parameter(torch.tensor(-1.2586))
        self.z0 = nn.Parameter(torch.zeros(d_lat), requires_grad=False)
        # ---- NEW: exposure path ----------------------------------------------------------
        # psi projects the SAME frozen item identity; no new 18k table (design sheet section 4).
        self.psi = nn.Sequential(nn.Linear(d_lat, h_expo), nn.GELU(), nn.Linear(h_expo, h_expo))
        self.rho_expo = nn.Sequential(nn.Linear(h_expo + 1, h_expo), nn.GELU(),
                                      nn.Linear(h_expo, d_lat))
        nn.init.zeros_(self.rho_expo[-1].weight); nn.init.zeros_(self.rho_expo[-1].bias)
        self.gate_e = nn.Parameter(torch.tensor(0.5413))      # g_e(0)=0: no unseen tokens -> no effect
        # ---- NEW: trainable per-item prior ----------------------------------------------
        self.delta_b = nn.Parameter(torch.zeros(ni))          # zero-init; frozen decoder untouched

    # ---------------------------------------------------------------- frozen anchor
    def native_z(self, ids, pad, lvs):
        """Frozen RecVAE encoder mean on binarised LIKED item tokens. Unseen tokens carry
        lvs == UNSEEN_LEVEL, so the `lvs >= LIKE_MIN_LEVEL` test would wrongly admit them --
        they are excluded EXPLICITLY here."""
        B = ids.shape[0]
        seen = lvs < UNSEEN_LEVEL
        like = (lvs >= LIKE_MIN_LEVEL) & seen & (~pad) & (ids < self.ni)
        x = torch.zeros((B, self.ni), dtype=torch.float32)
        rows = like.nonzero(as_tuple=True)
        x[rows[0], ids[rows]] = 1.0
        has = x.sum(-1) > 0
        out = torch.zeros((B, self.d_out), dtype=torch.float32)
        if bool(has.any()):
            with torch.no_grad():
                mu, _ = self._native[0].encoder(x[has], dropout_rate=0.0)
            out[has] = mu
        return out

    def anchor(self, n_tok):
        return 1.0 - torch.exp(-F.softplus(self.anchor_b) * n_tok)

    # ---------------------------------------------------------------- forward
    def forward(self, ids, vals, pad, lvs):
        unseen = (lvs >= UNSEEN_LEVEL) & (~pad)               # exposure tokens
        answered = (~pad) & (~unseen)                         # taste tokens

        if self.nc > 0:
            is_c = (ids >= self.ni) & (~pad)
            e = self.item_emb(torch.where(is_c, torch.zeros_like(ids), ids))
            if bool(is_c.any()):
                ce = self.concept_emb((ids - self.ni).clamp(min=0))
                e = torch.where(is_c.unsqueeze(-1), ce, e)
        else:
            e = self.item_emb(ids)

        # ---- taste branch: unseen positions are masked OUT, so this is i25 exactly ----
        lv_safe = lvs.clamp(max=NLEV - 1)                     # sentinel would index out of gamma/beta
        x = self.gamma(lv_safe) * e + self.beta(lv_safe)
        ph = self.phi(x) * answered.unsqueeze(-1).float()
        sp = ph.sum(1)
        nz = self.native_z(ids, pad, lvs)
        n_tok = answered.sum(-1, keepdim=True).float()        # ANSWERED count only
        a = self.anchor(n_tok)
        anz = a * nz
        delta = self.rho(torch.cat([sp, anz, n_tok.log1p(), a], dim=-1))
        g = 1.0 - torch.exp(-F.softplus(self.gate_a) * n_tok)
        z = anz + g * delta

        # ---- exposure branch: zero contribution when there are no unseen tokens ----
        m_tok = unseen.sum(-1, keepdim=True).float()
        if bool(unseen.any()):
            pe = self.psi(e) * unseen.unsqueeze(-1).float()
            se = pe.sum(1)
            de = self.rho_expo(torch.cat([se, m_tok.log1p()], dim=-1))
            g_e = 1.0 - torch.exp(-F.softplus(self.gate_e) * m_tok)
            z = z + g_e * de
        return z

    # ---------------------------------------------------------------- scoring
    def logits(self, z, Wd, bd):
        """The ONE place the per-item prior enters. Every scoring path must go through here so
        delta_b cannot be silently omitted by one caller and not another."""
        return z @ Wd.T + bd + self.delta_b


def build_i26(ni, src, args, log=print):
    """i26 counterpart of build_model's i25 branch. Same frozen-decoder contract."""
    n_conc = int(getattr(args, "n_concepts", 0)) if getattr(args, "concept_tokens", False) else 0
    enc = I26Encoder(ni, src, d_lat=args.t_latent, token_mode=args.token, n_concepts=n_conc)
    decoder = nn.Linear(args.t_latent, ni)
    with torch.no_grad():
        decoder.weight.copy_(src.decoder.weight)
        decoder.bias.copy_(src.decoder.bias)
        norms = src.decoder.weight.norm(dim=1).clamp_min(1e-8)
        enc.item_emb.weight.copy_(src.decoder.weight / norms.unsqueeze(1))
    enc.item_emb.weight.requires_grad_(False)
    decoder.weight.requires_grad_(False); decoder.bias.requires_grad_(False)
    log("[model] i26: decoder FROZEN RecVAE W+b; exposure branch + delta_b ZERO-INIT "
        "(bit-identical to i25 at step 0)")
    params = [p for p in list(enc.parameters()) + list(decoder.parameters()) if p.requires_grad]
    n_tr = sum(p.numel() for p in params)
    n_expo = sum(p.numel() for p in list(enc.psi.parameters()) + list(enc.rho_expo.parameters())) + 1
    log(f"[model] arch=i26 TRAINABLE={n_tr:,} (exposure branch {n_expo:,}, delta_b {ni:,})")
    return enc, decoder, params
