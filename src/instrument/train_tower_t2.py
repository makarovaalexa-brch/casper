"""train_tower_t2.py -- T2': GRADED-NATIVE pb2-class set-encoder tower on the CANONICAL Liang-25M split.

Purpose (Step-2 design sheet v2, `docs/design/DESIGN_SHEET_STEP2_BELIEF.md`): train the interview TOWER --
an order-invariant set encoder over (item, half-star-level) tokens -- on the canonical full-profile ruler
(data/ml-25m/proc/, `scripts/baselines/liang_split.py --data ml-25m`). The existing pb2 checkpoint CANNOT
be scored on this ruler (its train users overlap the new test cohort = leak); this file re-trains pb2-class
on the canonical partition. Emits experiments/baselines/ml25m_liang/tower_t2.json.

ARCHITECTURE MODES (--arch {i25, attn}; default i25 -- 2026-07-23 revision, evidence-backed by the
pre-cleanup i25_fold recipe that reached native-0.014; the from-scratch attention encoder's 0.29
plateau is the imitation gap that recipe avoided. FOLD_MASTER F7: set-transformer 3x rejected here):
  --arch i25 (DEFAULT): RESIDUAL fold on the frozen native RecVAE encoder.
      z = native_z + g(n_tok) * rho(concat(sumpool_phi, native_z, log1p(n_tok)))
      * native_z = FROZEN RecVAE encoder mean on the BINARIZED LIKED tokens of the input set
        (likes = level >= 7; dislikes NEVER enter the native encoder -- same rule as the old latent
        teacher). EMPTY-LIKES GUARD (documented): the RecVAE encoder L2-normalises its input with NO
        zero guard (x / ||x||, NaN on a zero row) => rows with no liked token get native_z = 0.
      * phi = per-token 2-layer MLP on the graded FiLM token (gamma(level)*e_i + beta(level),
        e_i = 200-d NORMALIZED frozen decoder row), SUM-pooled with padding zeroed. DISLIKES ARE IN
        the phi tokens -- that is where the dislike signal lives.
      * rho = 2-layer MLP, FINAL LAYER ZERO-INIT => z == native_z BIT-EXACTLY at init.
      * The g(0)=0 evidence gate is KEPT on the residual (documented interplay): with native_z(empty)=0
        it makes enc(empty) decode to the decoder bias at EVERY point in training, not just init.
      * The frozen native encoder is NOT registered in the module (checkpoints stay small); it is
        re-attached from RECVAE_CKPT best_state at build -- resume assumes that file is stable.
      * DECODER ARMS: (A, primary/default) FROZEN RecVAE decoder W+b -- restores G0-identity, logged;
        (B, --train_decoder) trainable at the slow LR group as in warm mode.
      * G0 note (documented): z = native_z + delta means full-profile score == native RecVAE IFF
        delta == 0; post-training delta != 0, so G0-strength criterion = tower >= native - CI (the
        residual is val-selected). Emitted in the JSON.
      * Loss = v3 unchanged (held-likes NLL + clamped w_neg dislike negatives); NO latent KD
        (lam pinned 0 -- the native encoder is inside the forward, not a loss target).
      * v4 ADVERSARIAL-REVIEW REPAIRS (2026-07-23): S1 ANCHOR GATE -- native_z is attenuated by a
        trainable a(n)=1-exp(-softplus(b)*n) (a(0)=0, ~1 by n~16; the native anchor is BELOW the pop
        floor at k<=2, probe 0.115 vs 0.1345) and a(n) is fed into rho; init identity is now
        z == a(n)*native_z. S2 NATIVE-SEEDED BEST -- the init model's val full@10 seeds `best` and the
        best checkpoint before epoch 1 (below-native G0 impossible). S5 DEDUP ASSERT at the fold
        boundary (sum-pool is duplicate-sensitive; hard fail).
      * LAUNCH GATES (S3/S4, review ruling): arm A (frozen decoder) may only be CERTIFIED if it passes
        the G3 flip test AND the ablate_binarized canary; arm B (--train_decoder) is the PRIMARY arm
        for dislike capability pending those gates.
  --arch attn: the previous pb2-class attention path, in one of THREE TEACHER MODES below.

TEACHER MODES for --arch attn (--teacher {warm_init, recvae, none}; default warm_init):
  --teacher warm_init (DEFAULT; pre-registered escalation after the dislike-separability probe FAILED
      2026-07-22 -- the frozen RecVAE decoder cannot separate dislike neighborhoods at acceptable cost):
      TRAINABLE-decoder T2'. Decoder W+b AND input item identities are INITIALIZED from the RecVAE ckpt
      but TRAINABLE (in the optimizer, normal weight decay). Latent KD is OFF (a different geometry will
      emerge; the teacher is REMOVED from the loss path -- the ckpt is init-only). Item identities are
      normalized ONCE at init then train freely; the separate norm scalar feature is dropped in this
      mode. Everything else = v3 verbatim: gamma-sign init, dislike negatives w_neg, evidence gate
      g(0)=0 with z0 frozen at 0 (intercept = the TRAINABLE bias, init from RecVAE's), G0 split
      reporting (the strength tie target stays the frozen RecVAE full-profile). The trunk-drift assert
      is REMOVED in this mode (things are supposed to move); it stays in --teacher recvae.
  --teacher recvae: distill T1 RecVAE into the set encoder with FROZEN geometry.
      * Loads `.cache/baselines/recvae_ml25m_liang.pt` (src/baselines/recvae.py state-dict; prefers
        state.best_state = the best-on-val weights, HR9, else the last `model`). d_latent=200.
      * FROZEN + reused: the RecVAE decoder Linear(200 -> n_items) INCLUDING its bias. The set encoder
        outputs 200-d (internal width D=512, projected out by the head).
      * FROZEN input item embeddings = the decoder rows Wd (n_items x 200), lifted to the internal width
        by a TRAINABLE in_proj Linear(200 -> 512). Fallback arm: --unfreeze_emb trains the embeddings.
      * Trainable params ONLY: attention blocks (mab_in / sab / PMA + inducing points), FiLM gamma/beta
        tables, in_proj, out head (normal init; gate guards the intercept) + z0. The FiLM tables REMAIN trainable by design: with the
        item embeddings frozen, gamma/beta are where graded value/sign lives.
      * SIGN INIT ON GAMMA [repair 2; --sign_prior default ON, --no_sign_prior ablation]: insurance
        against the RUNG1 gamma(hated)==gamma(loved) failure -- the sign is UNREACHABLE from a pure null
        init (memory `token-fusion-signed-values`). Init gamma(level) = v(level) in [-1,+1] (v = centered
        valence, NEUTRAL at the 3-star point, clipped), so a hated token STARTS as -e_i = PER-ITEM
        negation, a loved token as +e_i; beta ZERO-init (learned small). The earlier beta-along-u-bar
        variant is REMOVED: it seeded valence on the POPULARITY axis (G3a false-pass, G5 structurally
        unpassable).
      * TOKEN IDENTITY [repair 3]: input item identity = NORMALIZED decoder rows Wd[i]/||Wd[i]||, with
        ||Wd[i]|| appended as a SCALAR token feature (in_proj eats d_emb+1). The OUTPUT decoder stays the
        raw frozen Wd + bias -- normalization is input-side only (identity direction decoupled from the
        popularity-correlated row norm).
      * FREEZE INTEGRITY [repair 6]: frozen tensors are EXCLUDED from the optimizer param list entirely
        (AdamW weight-decay mutates even at zero grad) AND a per-epoch trunk-drift assert checks the
        frozen decoder W/b and item identities are bit-identical to the checkpoint.
      * LOSS (v3, post-adversarial-review 2026-07-22) = per-example
            lam_i * ||z_set(graded S'_i) - z_T_i||^2  +  (1 - lam_i) * rank_i
        rank_i = NLL(decoder(z_set), held likes_i) + w_neg * mean-logprob(held DISLIKES_i)   [repair 4:
        held disliked items (rating<=2.5) enter as explicit DOWN-WEIGHTED negatives -- without this,
        dislikes are input-only and get ~zero target-side gradient. --w_neg, default 0.1.]
        z_T = FROZEN RecVAE encoder MEAN on the LIKES-ONLY binarization of S' (review decision
        2026-07-22: only tokens with level >= 7, i.e. rating > 3.5, enter the teacher input -- RecVAE
        never saw dislikes as positives; a hated film fed as binary 1 would make z_T believe the user
        likes it. Dislikes STAY in the student tokens and the w_neg term. Rows whose subset has no
        liked token get ramp=0, i.e. no latent loss).
        lam_i = lambda_z SCHEDULE [repair 1], NOT flat: lam_i = anneal(epoch) * ramp(k_i) with
        ramp(k)=0 for k<=8 (the teacher latent is covariate-shift garbage at tiny k: RecVAE L2-normalises
        its input, putting 1-8-item subsets far outside training scale), linear ramp to 1 at k>=30; and
        anneal(ep) = lambda_z * max(0, 1 - ep/anneal_epochs) -- the latent KD is a full-profile
        WARM-START only; the NLL owns the endgame. Teacher forward SKIPPED for ramp==0 rows.
        Subset sizes sampled BROADLY: P_INTERVIEW=50% of examples draw k ~ U{1..8}; rest pb2 dropout.
      * EXACT EMPTY-SET INTERCEPT [repair 5]: evidence gate z = z0 + g(n_tok) * fold with
        g(n) = 1 - exp(-softplus(a)*n), so g(0)=0 EXACTLY and FOREVER (the RUNG1 intercept fix -- not an
        init-only property). z0 is FROZEN at 0 in teacher mode => enc(empty) decodes EXACTLY to
        softmax(RecVAE's trained bias) at every point in training. NOTE the intercept is RecVAE's LEARNED
        MARGINAL, not log-count popularity (no log-pop bias init in teacher mode; the bias is the ckpt's).
      * G0 REPORTING SPLIT [repair 7], emitted in tower_t2.json:
          G0-strength = CI-tie test of the tower's full-fold test NDCG@10 vs the frozen RecVAE's own
          full-profile NDCG@10 on the same ruler (|diff| <= 1.96*sqrt(se_a^2+se_b^2));
          G0-identity  = mechanical bit-identity checks (empty-set decode == frozen bias; the graded
          predict harness is a pure pass-through of the encoder fold).
  --teacher none: the previous from-scratch arm (trainable decoder, pop-log bias init, no latent teacher).

ARCHITECTURE PROVENANCE
  The model classes (MAB, PMA, SetEncoder) are COPIED, not imported, from the pinned reference
  `scripts/_verify/set_mn_pb2.py` (pin: set_mn_pb2.py @ 2bace5e; the Jul-14 frozen pb2 version of set_mn).
  Live `scripts/set_mn.py` is NOT imported -- it drifts (concept machinery, belief-pool, PrecAcc branches).
  STRIPPED from the copy: concept machinery, the belief-pool branch (pool="belief", lam_head/log_p0/
  pscale/last_prec), the a0c teacher / SignedAE warm-start, and the answerer. `load_answerer` is RETIRED
  (Jul-22 audit) and MUST NOT appear anywhere in this file -- asserted at import.
  Deviations from the pin (all revision-mandated): d_out decoupled from the internal width (head
  Linear(d+1 -> d_out), normal init -- the intercept identity lives in the g(0)=0 gate, not the init), optional frozen item_emb + in_proj lift.
  The RecVAE teacher is IMPORTED from src/baselines/recvae.py (the snap-certified port), not copied.

GRADED DATA (the point of "graded-native"): the proc CSVs store only binarised likes (>3.5). We re-derive
  half-star levels for ALL rating bands from the raw ratings.csv, for TRAIN-partition users only, mapped to
  the proc sid space. Training input tokens = (sid, level) over ALL bands (dislikes are levels too -- signed
  evidence); the multinomial target = the user's held-out LIKED items (>3.5), matching what the ruler scores.
  G3 CANARY (graded-vs-binarised ablation arm): --ablate_binarized collapses every INPUT token's level to a
  constant like-level (8 == 4.5 stars) so the encoder sees presence-only evidence; targets/eval unchanged.

  GRADED FOLD-IN CONVENTION (documented): the canonical eval path (metrics.evaluate) folds in the binarised
  tr-half LIKES. The graded encoder eats (sid,level) tokens, so we recover each tr-half item's REAL half-star
  level from the raw ratings and feed those. The fold-in ITEM SET is exactly the proc tr-half set (all likes,
  levels 7-9); we do NOT inject val/test dislikes (they were filtered out of the split -- injecting them would
  deviate from the canonical fold-in). The predict_fn wraps the encoder fold of these graded tokens and is
  scored through metrics.evaluate + head_mask, identical to run_ml25m_liang.

LEAK CHECKS (hard asserts, fail-fast):
  - vocab is EXACTLY the proc vocab (n_items == 18359; every training sid in [0, n_items)).
  - the reproduced TRAIN partition's binarised like-count == train.csv nnz (bit-faithful partition proof).
  - the reproduced train-user item vocabulary SET == unique_sid.txt SET (permutation reproduced correctly).
  - train users are disjoint from val+test users (no held-out user contributes any training row).

CHECKPOINT/RESUME (HR9/HR10): keeps <tag>.pt (last, for resume) and <tag>_best.pt (best-on-val only).
  Early stop on VAL full NDCG@10 via the canonical evaluate path. Code committed before any launch (HR10);
  this file does NOT launch training (review gate).

Usage:
  python src/instrument/train_tower_t2.py --smoke            # synthetic code-path smoke (fake teacher)
  python src/instrument/train_tower_t2.py --dry_run          # real-data leak asserts + 1 fwd/bwd + epoch est
  python src/instrument/train_tower_t2.py --train --tag t2   # FULL train (do NOT launch until review clears)
  python src/instrument/train_tower_t2.py --train --teacher none            # from-scratch fallback arm
  python src/instrument/train_tower_t2.py --train --alpha_kd 0.3 --teacher_npy <edlae_B.npy>  # EDLAE CE hook
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count()))
torch.set_num_threads(os.cpu_count())

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src", "baselines"))
sys.path.insert(0, os.path.join(_ROOT, "scripts", "baselines"))
import metrics as M                       # canonical loaders + evaluate + NDCG (vae_cf port)
import liang_split as LS                  # reuse the split's own filtering helpers (no drift)
import recvae as R                        # T1 teacher architecture (IMPORTED, snap-certified port)

PROC = os.path.join(_ROOT, "data", "ml-25m", "proc")
RAW = os.path.join(_ROOT, "data", "movielens", "ratings.csv")
OUTDIR = os.path.join(_ROOT, "experiments", "baselines", "ml25m_liang")
CKPT_DIR = os.path.join(_ROOT, ".cache", "instrument")
RECVAE_CKPT = os.path.join(_ROOT, ".cache", "baselines", "recvae_ml25m_liang.pt")

D = 512          # internal attention width (pb2)
NLEV = 10        # half-star levels 0.5..5.0 -> 0..9 (matches set_mn_pb2 gamma/beta tables)
LIKE_LEVEL = 8       # constant level (4.5 stars) used by the --ablate_binarized G3-canary arm
LIKE_MIN_LEVEL = 7   # like boundary: level >= 7 <=> star >= 4.0 <=> rating > 3.5 (Liang binarization)
NEG_FLOOR_OFFSET = 2.0  # dislike-term floor: log(1/n_items) - 2 (saturates ~e^2 below uniform)
COLD_SEED = 4242        # fixed RNG seed for the cold-val k-subsets (sampled ONCE, stable across epochs)
G0_BAR = 0.3540      # G0-strength bar on THIS ruler: frozen RecVAE reference full NDCG@10 (NOT the old
                     # 0.486 arena number -- different split, different fold-in protocol)
G0_TIE_CI = 0.0070   # paired-bootstrap CI half-width on the 10k cohort (measured at the G0 test eval);
                     # the --select_cold full-profile tie band: full@10 >= native_ref - G0_TIE_CI

# `load_answerer` is RETIRED (Jul-22 audit). Guard: this name must never be defined or called here.
assert "load_answerer" not in globals(), "load_answerer is retired and must not appear in the tower"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =============================================================================================
# ARCHITECTURE  (copied from scripts/_verify/set_mn_pb2.py @ pin 2bace5e; belief/concept/teacher stripped;
#                d_out decoupling + frozen-emb lift + gated fold = the 2026-07-22 revision)
# =============================================================================================
class MAB(nn.Module):
    """Multihead attention block (Set Transformer): MAB(Q,K) = LN(H + FF(H)), H = LN(Q + Attn(Q,K,K)).
    Verbatim from set_mn_pb2.py."""
    def __init__(self, d, nhead, drop=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, nhead, dropout=drop, batch_first=True)
        self.ln0 = nn.LayerNorm(d); self.ln1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * d, d))

    def forward(self, Q, K, key_padding_mask=None):
        h, _ = self.attn(Q, K, K, key_padding_mask=key_padding_mask, need_weights=False)
        h = self.ln0(Q + h)
        return self.ln1(h + self.ff(h))


class PMA(nn.Module):
    """Pooling by multihead attention: a learned seed attends over the set -> one vector.
    Verbatim from set_mn_pb2.py."""
    def __init__(self, d, nhead, drop=0.1):
        super().__init__()
        self.S = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.mab = MAB(d, nhead, drop)

    def forward(self, X, pad):
        b = X.shape[0]
        return self.mab(self.S.expand(b, -1, -1), X, key_padding_mask=pad)[:, 0]


class SetEncoder(nn.Module):
    """Graded-native set encoder (pb2-class, attn pool). Tokens (item, half-star level) -> [in_proj] ->
    FiLM token -> inducing-point attention (mab_in) -> global interaction (sab) -> PMA pool -> normal-init
    head -> z (d_out). Permutation-invariant, ANY set length.

    Copied from set_mn_pb2.SetEncoder (pool='attn', token='film'); belief-pool branch REMOVED. Revisions
    (2026-07-22, v3 post-adversarial-review): d_out decoupled from the internal width d; item_emb may be
    FROZEN to NORMALIZED teacher decoder rows with ||Wd[i]|| as an appended scalar feature (repair 3);
    EVIDENCE GATE z = z0 + g(n_tok)*fold, g(n)=1-exp(-softplus(a)*n) => g(0)=0 EXACTLY, so the empty set
    decodes to decoder(z0) forever, not just at init (repair 5; z0 frozen at 0 in teacher mode)."""
    def __init__(self, ni, d=D, d_out=D, d_emb=None, nhead=4, m=32, nlayers=2, token_mode="film",
                 norm_feat=False):
        super().__init__()
        d_emb = d_emb or d
        self.ni = ni; self.d = d; self.d_out = d_out; self.token_mode = token_mode
        self.norm_feat = norm_feat
        self.item_emb = nn.Embedding(ni, d_emb)
        nn.init.normal_(self.item_emb.weight, std=0.02)     # overwritten+frozen in teacher mode
        self.register_buffer("emb_norm", torch.ones(ni))    # ||Wd[i]|| scalar feature (repair 3)
        d_in = d_emb + (1 if norm_feat else 0)
        self.in_proj = nn.Linear(d_in, d) if d_in != d else nn.Identity()
        # FiLM graded token: per-level scale+shift tables. init gamma=1, beta~small -> token = e + beta
        # (the pure additive/author-null token); the model can LEARN gamma<0 for dislike-as-negation.
        self.gamma = nn.Embedding(NLEV, d); self.beta = nn.Embedding(NLEV, d)
        nn.init.ones_(self.gamma.weight); nn.init.normal_(self.beta.weight, std=0.02)
        # mlp token path kept for parity (continuous signed value); FiLM is the graded-native default.
        self.tok_mlp = nn.Sequential(nn.Linear(d + 1, d), nn.GELU(), nn.Linear(d, d))
        self.I = nn.Parameter(torch.randn(1, m, d) * 0.02)             # inducing points
        self.mab_in = MAB(d, nhead)                                    # I attends to the set  (O(L*m))
        self.sab = nn.ModuleList([MAB(d, nhead) for _ in range(nlayers)])   # interaction among inducing pts
        self.pma = PMA(d, nhead)                                       # pool -> one vector
        self.z0 = nn.Parameter(torch.zeros(d_out))                     # empty-set prior mean (0)
        self.head = nn.Linear(d + 1, d_out)                            # + log(1+set-size) feature
        # UN-ZEROED head (archaeology verdict 2026-07-23): normal init -- the fold is LIVE from step 0
        # (no pure cold-start tax). The empty-set intercept does NOT depend on head init: the gate
        # g(0)=0 zeroes the head output at n_tok=0 exactly, at every point in training.
        nn.init.normal_(self.head.weight, std=0.02); nn.init.normal_(self.head.bias, std=0.02)
        self.gate_a = nn.Parameter(torch.tensor(0.5413))               # softplus(0.5413) ~= 1.0

    def forward(self, ids, vals, pad, lvs):
        b = ids.shape[0]
        e_in = self.item_emb(ids)
        if self.norm_feat:                                             # repair 3: [Wd_i/||Wd_i||, ||Wd_i||]
            e_in = torch.cat([e_in, self.emb_norm[ids].unsqueeze(-1)], dim=-1)
        e = self.in_proj(e_in)
        if self.token_mode == "film":
            x = self.gamma(lvs) * e + self.beta(lvs)                                   # (B,L,d) CHEAP
        else:
            x = self.tok_mlp(torch.cat([e, vals.unsqueeze(-1)], dim=-1))               # (B,L,d)
        h = self.mab_in(self.I.expand(b, -1, -1), x, key_padding_mask=pad)            # (B,m,d)
        for blk in self.sab:
            h = blk(h, h)                                                             # global interaction
        p = self.pma(h, None)                                                         # (B,d)
        p = torch.nan_to_num(p)             # all-padded row -> softmax over -inf -> NaN; empty set -> z0
        n_tok = (~pad).sum(-1, keepdim=True).float()
        sz = n_tok.clamp_min(1.0).log1p()
        fold = self.head(torch.cat([p, sz], dim=-1))
        g = 1.0 - torch.exp(-F.softplus(self.gate_a) * n_tok)          # repair 5: g(0)=0 EXACTLY, always
        return self.z0 + g * fold


class I25Encoder(nn.Module):
    """RESIDUAL fold on the frozen native RecVAE encoder (--arch i25; 2026-07-23 revision):
        z = native_z + g(n_tok) * rho(concat(sum_phi, native_z, log1p(n_tok)))
    native_z = frozen RecVAE encoder mean on the BINARIZED LIKES of the token set (level >= 7;
    rows with NO liked token get native_z = 0 -- the RecVAE encoder x/||x|| has no zero guard).
    phi = 2-layer MLP on the graded FiLM token (dislikes ARE here), SUM-pooled (padding zeroed).
    rho final layer ZERO-INIT => z == native_z bit-exactly at init. No PMA/MAB (FOLD_MASTER F7).
    The native encoder is held UNREGISTERED (self._native list) so checkpoints exclude its ~13M
    params; build_model re-attaches it from RECVAE_CKPT best_state."""
    def __init__(self, ni, native_encoder, d_lat=200, h=512, token_mode="film", n_concepts=0):
        super().__init__()
        self.ni = ni; self.d = d_lat; self.d_out = d_lat; self.token_mode = token_mode
        self.norm_feat = False
        self._native = [native_encoder]                      # UNREGISTERED (frozen, external)
        self.item_emb = nn.Embedding(ni, d_lat)              # frozen normalized decoder rows
        # ARM C-FULL (--concept_tokens, 2026-07-24): OPTIONAL concept-token channel. Token id ni+c is a
        # CONCEPT token: identity = concept_emb[c] (TRAINABLE, whitened-centroid init applied in train()),
        # fused with its graded value through the SAME FiLM gamma/beta tables as item tokens, summed into
        # the SAME phi pool. Concept tokens NEVER enter the frozen native RecVAE anchor (item-likes only)
        # -- that is the G0 protection: an item-only input bypasses this channel bit-exactly.
        self.nc = n_concepts
        if n_concepts > 0:
            self.concept_emb = nn.Embedding(n_concepts, d_lat)
            nn.init.normal_(self.concept_emb.weight, std=0.02)   # overwritten w/ whitened centroids
        self.gamma = nn.Embedding(NLEV, d_lat); self.beta = nn.Embedding(NLEV, d_lat)
        nn.init.ones_(self.gamma.weight); nn.init.normal_(self.beta.weight, std=0.02)
        self.phi = nn.Sequential(nn.Linear(d_lat, h), nn.GELU(), nn.Linear(h, h))
        # rho input: [sum_phi, a(n)*native_z, log1p(n), a(n)]  (S1: a(n) passed into rho)
        self.rho = nn.Sequential(nn.Linear(h + d_lat + 2, h), nn.GELU(), nn.Linear(h, d_lat))
        nn.init.zeros_(self.rho[-1].weight); nn.init.zeros_(self.rho[-1].bias)   # z==a*native at init
        self.gate_a = nn.Parameter(torch.tensor(0.5413))     # g(0)=0: empty set -> 0
        # S1 ANCHOR GATE (v4 adversarial review): a(n) = 1 - exp(-softplus(b)*n). The native anchor is
        # BELOW the pop floor at k<=2 (probe 0.115 vs 0.1345) -> attenuate it where it is wrong.
        # b init: softplus(b) ~= 0.25 -> a(16) ~= 0.98 (ramps to ~1 by n~16); TRAINABLE scalar.
        self.anchor_b = nn.Parameter(torch.tensor(-1.2586))  # softplus(-1.2586) ~= 0.25
        self.z0 = nn.Parameter(torch.zeros(d_lat), requires_grad=False)          # compat (unused)

    def native_z(self, ids, pad, lvs):
        """Frozen RecVAE encoder mean on binarized LIKED ITEM tokens; 0 for rows with no likes.
        Concept tokens (ids >= ni) are EXCLUDED unconditionally -- the frozen anchor never sees them."""
        B = ids.shape[0]
        like = (lvs >= LIKE_MIN_LEVEL) & (~pad) & (ids < self.ni)
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
        """S1 anchor gate a(n) = 1 - exp(-softplus(b)*n); a(0)=0 exactly, ramps to ~1 by n~16."""
        return 1.0 - torch.exp(-F.softplus(self.anchor_b) * n_tok)

    def forward(self, ids, vals, pad, lvs):
        if self.nc > 0:
            is_c = (ids >= self.ni) & (~pad)                                 # concept-token positions
            e = self.item_emb(torch.where(is_c, torch.zeros_like(ids), ids))
            if bool(is_c.any()):                                             # no concepts -> path bit-
                ce = self.concept_emb((ids - self.ni).clamp(min=0))          # identical to nc=0 build
                e = torch.where(is_c.unsqueeze(-1), ce, e)
        else:
            e = self.item_emb(ids)                                           # (B,L,200) frozen
        x = self.gamma(lvs) * e + self.beta(lvs)                             # graded FiLM token
        ph = self.phi(x) * (~pad).unsqueeze(-1).float()                      # padding zeroed
        sp = ph.sum(1)                                                       # (B,h) SUM pool
        nz = self.native_z(ids, pad, lvs)                                    # (B,200) frozen fold
        n_tok = (~pad).sum(-1, keepdim=True).float()
        a = self.anchor(n_tok)                                               # (B,1) S1 anchor gate
        anz = a * nz                                                         # attenuated native anchor
        delta = self.rho(torch.cat([sp, anz, n_tok.log1p(), a], dim=-1))
        g = 1.0 - torch.exp(-F.softplus(self.gate_a) * n_tok)               # g(0)=0 exactly
        return anz + g * delta                                               # empty: a(0)*0 + 0 = 0


# =============================================================================================
# T1 RECVAE TEACHER  (frozen; encoder mean supervises the latent, decoder+bias reused for ranking)
# =============================================================================================
def load_recvae_teacher(ni, path=RECVAE_CKPT, hidden=600, latent=200):
    """Load the T1 RecVAE (src/baselines/recvae.py checkpoint {'model','state',...}); prefer the
    best-on-val weights (state.best_state, HR9), else the last 'model'. Returns frozen eval model."""
    if not os.path.exists(path):
        raise SystemExit(f"[teacher] RecVAE checkpoint not found: {path} (still training?). "
                         f"Use --teacher none for the from-scratch arm.")
    blob = torch.load(path, map_location="cpu")
    st = blob.get("state", {})
    sd = st.get("best_state") or blob["model"]
    src = "state.best_state" if st.get("best_state") else "model (last)"
    t = R.RecVAE(hidden, latent, ni)
    t.load_state_dict(sd)
    t.eval()
    for p in t.parameters():
        p.requires_grad_(False)
    log(f"[teacher] RecVAE loaded from {src}: ep{st.get('epoch','?')} best_val={st.get('best', float('nan')):.4f} "
        f"(hidden={hidden} latent={latent}) -- FROZEN")
    return t


def teacher_latent(teacher, sids_list, ni):
    """z_T = frozen RecVAE encoder MEAN on the BINARIZED subset (presence of ALL subset tokens ->
    1.0; the encoder L2-normalises internally; subsets are never empty by construction).
    sids_list: list of int arrays. Returns (B, latent)."""
    x = torch.zeros((len(sids_list), ni), dtype=torch.float32)
    for r, s in enumerate(sids_list):
        x[r, s] = 1.0
    with torch.no_grad():
        mu, _ = teacher.encoder(x, dropout_rate=0.0)
    return mu


# =============================================================================================
# DATA  (graded profiles from raw stars; canonical partition reproduced via liang_split helpers)
# =============================================================================================
def star_to_level(rating):
    """half-star rating (0.5..5.0) -> discrete level 0..9 (NLEV=10)."""
    return np.clip(np.rint(np.asarray(rating, np.float64) * 2).astype(np.int64) - 1, 0, NLEV - 1)


def level_to_sv(level):
    """level 0..9 -> signed value (scale_rating inverse: star = sv*2.25+2.75 => sv=(star-2.75)/2.25).
    Only used by the mlp token path; FiLM ignores vals."""
    star = (np.asarray(level, np.float64) + 1.0) / 2.0
    return ((star - 2.75) / 2.25).astype(np.float32)


def reproduce_partition():
    """Reproduce the canonical Liang-25M user partition (which raw userIds are train/val/test) EXACTLY,
    using liang_split's own helpers + constants (no drift). Returns (unique_uid, tr_set, vd_set, te_set,
    n_train, raw_catalog_df, show2id, unique_sid_list). raw_catalog_df = catalog-restricted, ALL bands."""
    import pandas as pd
    log(f"[data] reading raw ratings {RAW}")
    raw = pd.read_csv(RAW)
    # catalog restriction (author-approved deviation; must be exactly 18430) -- as in liang_split.py
    icnt = raw.groupby("movieId").size()
    catalog = set(icnt[icnt >= LS.CATALOG_MIN_RATINGS].index.tolist())
    if len(catalog) != LS.CATALOG_N_EXPECTED:
        raise SystemExit(f"[data] CATALOG MISMATCH: {len(catalog)} != {LS.CATALOG_N_EXPECTED}; STOP.")
    raw = raw[raw["movieId"].isin(catalog)].copy()          # catalog-restricted, ALL rating bands (graded)
    # binarise + filter to reproduce the user set + permutation (identical to liang_split.main)
    binr = raw[raw["rating"] > LS.RATING_GT]
    binr, user_activity, _ = LS.filter_triplets(binr)
    unique_uid = user_activity["userId"].values
    np.random.seed(LS.SEED)
    unique_uid = unique_uid[np.random.permutation(unique_uid.size)]
    n = unique_uid.size
    n_heldout = LS.N_HELDOUT
    tr_users = unique_uid[:(n - n_heldout * 2)]
    vd_users = unique_uid[(n - n_heldout * 2):(n - n_heldout)]
    te_users = unique_uid[(n - n_heldout):]
    # authoritative sid order = unique_sid.txt (guarantees sid alignment with the trained baselines)
    unique_sid_list = [int(x) for x in open(os.path.join(PROC, "unique_sid.txt")).read().split()]
    show2id = {sid: i for i, sid in enumerate(unique_sid_list)}
    # ---- LEAK / FIDELITY ASSERTS ----
    meta = M.load_meta(PROC)
    assert len(unique_sid_list) == meta["n_items"] == 18359, "vocab size != proc vocab"
    assert len(tr_users) == meta["n_train_users"] == 140768, f"train users {len(tr_users)} != 140768"
    assert len(vd_users) == len(te_users) == 10000, "held-out user count != 10000"
    tr_set, vd_set, te_set = set(tr_users.tolist()), set(vd_users.tolist()), set(te_users.tolist())
    assert tr_set.isdisjoint(vd_set) and tr_set.isdisjoint(te_set), "train overlaps held-out (LEAK)"
    assert vd_set.isdisjoint(te_set), "val overlaps test"
    # reproduced train vocab SET must equal unique_sid.txt SET (proves the permutation was reproduced)
    train_plays = binr[binr["userId"].isin(tr_set)]
    repro_vocab = set(pd.unique(train_plays["movieId"]).tolist())
    assert repro_vocab == set(unique_sid_list), "reproduced train vocab != unique_sid.txt (partition drift)"
    # reproduced binarised train like-count must equal train.csv nnz (bit-faithful partition proof)
    n_like_train = int(train_plays[train_plays["movieId"].isin(show2id)].shape[0])
    train_csv_nnz = int(M.load_train(meta["n_items"], PROC).nnz)
    assert n_like_train == train_csv_nnz, f"train like-count {n_like_train} != train.csv nnz {train_csv_nnz}"
    log(f"[data] LEAK CHECKS PASS: vocab=18359, train=140768 disjoint from 2x10000 held-out, "
        f"train likes={n_like_train}==train.csv nnz, vocab-set matches unique_sid.txt")
    return unique_uid, tr_set, vd_set, te_set, len(tr_users), raw, show2id, unique_sid_list


def build_train_profiles(raw, tr_set, show2id, max_users=None):
    """Graded profiles for TRAIN-partition users: {items(sid,all bands), levels, vals(signed), liked(sid)}.
    max_users caps the number of users MATERIALISED (dry-run only) -- NOT a data cap on a real run."""
    import pandas as pd
    df = raw[raw["userId"].isin(tr_set)].copy()
    df["sid"] = df["movieId"].map(show2id)
    df = df[df["sid"].notna()]                              # drop out-of-vocab items
    df["sid"] = df["sid"].astype(np.int64)
    df["lvl"] = star_to_level(df["rating"].values)
    df["like"] = (df["rating"].values > LS.RATING_GT)
    df["dislike"] = (df["rating"].values <= 2.5)            # repair 4: explicit dislike band (<=2.5 stars)
    users = []
    uids = df["userId"].unique()
    if max_users is not None:
        uids = uids[:max_users]
        df = df[df["userId"].isin(set(uids.tolist()))]
    for _, g in df.groupby("userId"):
        sids = g["sid"].values.astype(np.int64)
        lvls = g["lvl"].values.astype(np.int64)
        liked = g.loc[g["like"], "sid"].values.astype(np.int64)
        disliked = g.loc[g["dislike"], "sid"].values.astype(np.int64)
        if len(liked) == 0:
            continue                                        # unique_uid filter guarantees >=5, but be safe
        users.append({"items": sids, "levels": lvls,
                      "vals": level_to_sv(lvls), "liked": liked, "disliked": disliked})
    log(f"[data] materialised {len(users)} train graded profiles"
        + (f" (capped at {max_users})" if max_users is not None else ""))
    return users


def build_graded_eval_matrix(raw, unique_uid, show2id, unique_sid_list, split):
    """Build a level+1 CSR aligned to the proc {split}_tr fold-in matrix (same shape/row order), by
    recovering each tr-half item's REAL half-star level from raw. Returns (L_csr, n_users). Integrity:
    every proc tr pair must exist in raw (asserted) and L.nnz == {split}_tr.csv nnz."""
    import pandas as pd
    from scipy import sparse
    meta = M.load_meta(PROC); ni = meta["n_items"]
    tr = pd.read_csv(os.path.join(PROC, f"{split}_tr.csv"))
    uid = tr["uid"].values; sid = tr["sid"].values
    userId = np.asarray(unique_uid)[uid]
    movieId = np.asarray(unique_sid_list)[sid]
    key = pd.DataFrame({"userId": userId, "movieId": movieId, "sid": sid, "uid": uid})
    merged = key.merge(raw[["userId", "movieId", "rating"]], on=["userId", "movieId"], how="left")
    assert not merged["rating"].isna().any(), f"{split}_tr has (user,item) pairs absent from raw (drift)"
    lvl = star_to_level(merged["rating"].values)
    start_idx = int(uid.min())                              # matches metrics._load_tr_te_data reindex
    rows = merged["uid"].values - start_idx
    n_users = int(uid.max()) - start_idx + 1
    L = sparse.csr_matrix((lvl.astype(np.float32) + 1.0, (rows, merged["sid"].values)),
                          dtype=np.float32, shape=(n_users, ni))
    assert L.nnz == len(tr), f"{split} graded matrix nnz {L.nnz} != {split}_tr rows {len(tr)}"
    return L, n_users


# =============================================================================================
# BATCHING  (length-bucketed; copied contract from set_mn_pb2 -- HARD RULE #1: bucket, never cap)
# =============================================================================================
ATTN_BUDGET = 1_000_000
MAX_B = 256


def make_batches(users, order):
    batches = []; cur = []; curL = 0
    for i in order:
        L = len(users[i]["items"])
        nl = max(curL, L)
        if cur and ((len(cur) + 1) * nl > ATTN_BUDGET or len(cur) >= MAX_B):
            batches.append(cur); cur = []; curL = 0; nl = L
        cur.append(i); curL = nl
    if cur:
        batches.append(cur)
    return batches


P_INTERVIEW = 0.5      # default fraction of examples drawn in the interview regime (see --p_interview)
INTERVIEW_KMAX = 8


def parse_p_interview(spec):
    """--p_interview '0.5' -> (0.5, 0.5); '0.1:0.5' -> linear 0.1 -> 0.5 over anneal_epochs."""
    parts = str(spec).split(":")
    p0 = float(parts[0]); p1 = float(parts[1]) if len(parts) > 1 else p0
    return p0, p1


def p_interview_at(ep, p0, p1, anneal_epochs):
    return p0 + (p1 - p0) * min(1.0, ep / max(anneal_epochs, 1))


def make_input_target(u, rng, drop_max=0.5, p_int=P_INTERVIEW):
    """Denoising curriculum, revision 2026-07-22: with prob p_int sample an INTERVIEW-REGIME subset
    (k ~ U{1..8} tokens -- supervises the small-set fold the interview lives in); otherwise the pb2
    random-dropout subset. Input includes dislikes (graded); target = liked NOT in input (leak-free)."""
    its = u["items"]; n = len(its)
    if rng.random() < p_int:
        k = int(rng.integers(1, min(INTERVIEW_KMAX, n) + 1))
        keep = np.zeros(n, bool); keep[rng.choice(n, size=k, replace=False)] = True
    else:
        keep = rng.random(n) >= rng.uniform(0.0, drop_max)
        if not keep.any():
            keep[rng.integers(0, n)] = True
    inp = its[keep]; lv = u["levels"][keep]; sv = u["vals"][keep]
    tgt = np.setdiff1d(u["liked"], inp, assume_unique=False)
    if len(tgt) == 0:
        return None
    negs = np.setdiff1d(u.get("disliked", np.empty(0, np.int64)), inp, assume_unique=False)  # repair 4
    return inp, lv, sv, tgt, negs


def sel_value_to_level(v):
    """SEL-graded concept value v in [0.25, 1] -> FiLM level in the positive band {6..9}
    (0.25 -> 6 weak-positive, 1.0 -> 9 loved). Concepts carry no dislike band pre-C3 (SEL is
    watch-lift, positive-only); the mapping is documented as the Arm C-full convention."""
    v = min(max(float(v), 0.25), 1.0)
    return int(6 + round(3.0 * (v - 0.25) / 0.75))


def make_concept_example_signed(u, rng, Mm, pexp, member_sets, ni, item_mean, prereg,
                                p_conc_only=0.35, m_max=16):
    """SIGNED C-full curriculum example (DESIGN_SIGNED_CONCEPTS): four-band SEL+VAL labels from the
    input pool; concept token levels via signed_answers.value_to_level (dislikes -> the tower's
    EXISTING graded dislike levels); m ~ U{1..16}; C_NEG cap over the revealed negatives."""
    from signed_answers import signed_for_pool, cap_negatives, value_to_level, BAND_REFUSE
    its = np.asarray(u["items"], np.int64)
    liked = np.asarray(u["liked"], np.int64)
    if len(its) < 4 or len(liked) < 2:
        return None
    ntg = max(1, len(liked) // 3)
    tg = rng.choice(liked, size=ntg, replace=False)
    keep = ~np.isin(its, tg)
    pool_s = its[keep]; pool_l = np.asarray(u["levels"], np.int64)[keep]
    pool_v = np.asarray(u["vals"], np.float32)[keep]
    if len(pool_s) < 2:
        return None
    stars = (pool_l.astype(np.float32) + 1.0) / 2.0
    V, B, ans = signed_for_pool(pool_s, stars, item_mean, Mm, pexp, prereg)
    cand = np.flatnonzero(ans & (B != BAND_REFUSE))
    if len(cand) == 0:
        return None
    m = min(int(rng.integers(1, m_max + 1)), len(cand))
    order = cand[np.argsort(-np.abs(V[cand]))]
    top = order[: int(np.ceil(m / 2))]
    rest_pool = np.setdiff1d(cand, top)
    rest = rng.choice(rest_pool, size=min(m - len(top), len(rest_pool)), replace=False) \
        if len(rest_pool) and m > len(top) else np.empty(0, np.int64)
    cids = np.concatenate([top, rest]).astype(int)
    cvals = cap_negatives([float(V[c]) for c in cids])
    support = [np.intersect1d(pool_s, member_sets[int(c)]) for c in cids]
    drop = np.unique(np.concatenate(support)) if support else np.empty(0, np.int64)
    mk = ~np.isin(pool_s, drop)
    rem_s, rem_l, rem_v = pool_s[mk], pool_l[mk], pool_v[mk]
    if rng.random() < p_conc_only or len(rem_s) == 0:
        in_s = np.empty(0, np.int64); in_l = np.empty(0, np.int64); in_v = np.empty(0, np.float32)
    else:
        k = int(rng.integers(1, min(8, len(rem_s)) + 1))
        pick = rng.choice(len(rem_s), size=k, replace=False)
        in_s, in_l, in_v = rem_s[pick], rem_l[pick], rem_v[pick]
    c_ids = np.asarray([ni + int(c) for c in cids], np.int64)
    c_lv = np.asarray([value_to_level(v) for v in cvals], np.int64)
    inp = np.concatenate([in_s, c_ids]); lvo = np.concatenate([in_l, c_lv])
    sv = np.concatenate([in_v, np.zeros(len(cids), np.float32)])
    tgt = np.setdiff1d(tg, in_s)
    if len(tgt) == 0:
        return None
    negs = np.setdiff1d(u.get("disliked", np.empty(0, np.int64)), in_s)
    return inp, lvo, sv, tgt, negs


def make_concept_example(u, rng, Mm, grate, member_sets, ni, p_conc_only=0.35, m_max=8):
    """ARM C-FULL item-masked curriculum example (mirrors concept_fold.make_example; the C-lite recipe):
    pool/target split (leak-free), SEL labels FROM THE INPUT POOL ONLY, the member items that generated
    each revealed concept DROPPED from the item input, item budget k (concepts-only w.p. p_conc_only),
    concept tokens appended as ids ni+c at sel_value_to_level(v). Returns the make_input_target tuple
    shape (inp, lv, sv, tgt, negs) or None."""
    from concept_fold import sel_concepts_from_items          # lazy (concept_fold imports this module)
    its = np.asarray(u["items"], np.int64); n = len(its)
    liked = np.asarray(u["liked"], np.int64)
    if n < 4 or len(liked) < 2:
        return None
    ntg = max(1, len(liked) // 3)
    tg = rng.choice(liked, size=ntg, replace=False)
    keep = ~np.isin(its, tg)
    pool_s = its[keep]; pool_l = np.asarray(u["levels"], np.int64)[keep]
    pool_v = np.asarray(u["vals"], np.float32)[keep]
    if len(pool_s) < 2:
        return None
    m = int(rng.integers(1, m_max + 1))
    cids, cvals, support = sel_concepts_from_items(pool_s, Mm, grate, m, member_sets)
    if not cids:
        return None
    dropset = np.unique(np.concatenate(support)) if support else np.empty(0, np.int64)
    mk = ~np.isin(pool_s, dropset)                            # ITEM MASK: independent-signal curriculum
    rem_s, rem_l, rem_v = pool_s[mk], pool_l[mk], pool_v[mk]
    if rng.random() < p_conc_only or len(rem_s) == 0:
        in_s = np.empty(0, np.int64); in_l = np.empty(0, np.int64); in_v = np.empty(0, np.float32)
    else:
        k = int(rng.integers(1, min(8, len(rem_s)) + 1))
        pick = rng.choice(len(rem_s), size=k, replace=False)
        in_s, in_l, in_v = rem_s[pick], rem_l[pick], rem_v[pick]
    c_ids = np.asarray([ni + c for c in cids], np.int64)
    c_lv = np.asarray([sel_value_to_level(v) for v in cvals], np.int64)
    inp = np.concatenate([in_s, c_ids]); lv = np.concatenate([in_l, c_lv])
    sv = np.concatenate([in_v, np.zeros(len(cids), np.float32)])     # concept sv=0 (value in the LEVEL)
    tgt = np.setdiff1d(tg, in_s)
    if len(tgt) == 0:
        return None
    negs = np.setdiff1d(u.get("disliked", np.empty(0, np.int64)), in_s)
    return inp, lv, sv, tgt, negs


def pack_tokens(rows, binarize=False):
    """rows: list of (ids, levels, vals) -> padded torch tensors (ids, vals, pad, lvs).
    binarize=True is the G3-canary arm: every input level collapsed to LIKE_LEVEL (presence-only)."""
    B = len(rows); L = max(1, max(len(r[0]) for r in rows))   # floor 1: an all-pad row is the canonical
    # empty-set path (attention NaN -> nan_to_num, gate g(0)=0 -> z0); L=0 would crash MultiheadAttention
    ids = np.zeros((B, L), np.int64); vals = np.zeros((B, L), np.float32)
    pad = np.ones((B, L), bool); lvs = np.zeros((B, L), np.int64)
    for r, (i, lv, sv) in enumerate(rows):
        k = len(i)
        # S5 dedup assert (fold boundary, HARD FAIL): sum-pool is duplicate-sensitive; a repeated sid
        # would silently double-count evidence. MovieLens guarantees unique (user,item), so this must
        # never fire -- if it does, the data path upstream is broken.
        assert len(np.unique(i)) == k, f"duplicate sids in a token set (row {r}: {k} tokens)"
        ids[r, :k] = i; pad[r, :k] = False
        if binarize:
            lvs[r, :k] = LIKE_LEVEL; vals[r, :k] = level_to_sv(np.full(k, LIKE_LEVEL))
        else:
            lvs[r, :k] = lv; vals[r, :k] = sv
    return (torch.from_numpy(ids), torch.from_numpy(vals), torch.from_numpy(pad), torch.from_numpy(lvs))


# =============================================================================================
# EVAL  (canonical metrics.evaluate path; predict_fn wraps the encoder fold of graded tokens)
# =============================================================================================
def truncate_graded(L, k, seed):
    """COLD-VAL fold-in: per user keep a FIXED random k-subset of their graded tokens (RandomState(seed),
    sampled once at build -- stable across epochs). Users with <=k tokens keep all. Returns a new CSR
    aligned row-for-row with L."""
    from scipy import sparse
    rng = np.random.RandomState(seed)
    rows, cols, data = [], [], []
    for r in range(L.shape[0]):
        s, e = L.indptr[r], L.indptr[r + 1]
        idx = np.arange(s, e)
        if len(idx) > k:
            idx = rng.choice(idx, size=k, replace=False)
        rows.extend([r] * len(idx)); cols.extend(L.indices[idx]); data.extend(L.data[idx])
    return sparse.csr_matrix((np.asarray(data, np.float32), (rows, cols)),
                             shape=L.shape, dtype=np.float32)


def make_graded_predict_fn(enc, Wd, bd, L_csr, binarize=False, check_nnz=True):
    """Factory: returns a predict_fn(X_csr)->dense scores for metrics.evaluate. The graded levels come
    from L_csr (aligned to the fold-in matrix); metrics.evaluate iterates rows sequentially so a cursor
    tracks the row offset. An nnz assert catches any misalignment (check_nnz=False for COLD eval, where
    L_csr is deliberately a truncated subset of the full fold-in). FRESH factory call per evaluate()."""
    state = {"pos": 0}

    def predict(X_csr):
        pos = state["pos"]; rows = X_csr.shape[0]
        Ls = L_csr[pos:pos + rows]; state["pos"] = pos + rows
        assert Ls.shape[0] == rows, "graded/fold-in row misalignment"
        assert (not check_nnz) or Ls.nnz == X_csr.nnz, "graded/fold-in misalignment"
        enc.eval(); out = np.zeros((rows, enc.ni), dtype=np.float32)
        with torch.no_grad():
            b = 0
            while b < rows:
                chunk = list(range(b, min(b + 256, rows)))
                packrows = []
                for r in chunk:
                    s, e = Ls.indptr[r], Ls.indptr[r + 1]
                    sids = Ls.indices[s:e].astype(np.int64)
                    lv = (Ls.data[s:e] - 1.0).astype(np.int64)
                    packrows.append((sids, lv, level_to_sv(lv)))
                ids, vals, pad, lvs = pack_tokens(packrows, binarize=binarize)
                z = enc(ids, vals, pad, lvs)
                out[chunk] = (z @ Wd.T + bd).numpy().astype(np.float32)
                b += len(chunk)
        return out
    return predict


def eval_split(enc, Wd, bd, raw, unique_uid, show2id, unique_sid_list, split, head_mask,
               batch_size=500, binarize=False):
    """NDCG@10 full/tail + NDCG@100 on a split via the canonical evaluate path."""
    meta = M.load_meta(PROC); ni = meta["n_items"]
    L, _ = build_graded_eval_matrix(raw, unique_uid, show2id, unique_sid_list, split)
    if split == "validation":
        d_tr, d_te = M.load_val(ni, PROC)
    else:
        d_tr, d_te = M.load_test(ni, PROC)
    predict = make_graded_predict_fn(enc, Wd, bd, L, binarize=binarize)
    return M.evaluate(predict, d_tr, d_te, batch_size=batch_size, head_mask=head_mask)


def cold_full10(enc, Wd, bd, L_k, d_tr, d_te, head_mask, binarize=False):
    """COLD val full@10: the encoder folds ONLY the fixed k-subset tokens, but the candidate mask and
    targets stay the FULL canonical ones (metrics.evaluate masks the full tr fold-in), so cold numbers
    are directly comparable to the full-fold val@10."""
    pred = make_graded_predict_fn(enc, Wd, bd, L_k, binarize=binarize, check_nnz=False)
    return M.evaluate(pred, d_tr, d_te, batch_size=500, head_mask=head_mask)["ndcg@10"]


def compute_head_mask(train, n_items):
    """HEAD = smallest item set covering 33% of TRAIN mass (identical to run_ml25m_liang)."""
    cnt = np.asarray(train.sum(axis=0)).ravel().astype(np.float64)
    order_pop = np.argsort(-cnt)
    cumfrac = np.cumsum(cnt[order_pop]) / cnt.sum()
    head = np.zeros(n_items, dtype=bool)
    head[order_pop[:np.searchsorted(cumfrac, 0.33) + 1]] = True
    return head, cnt


def report_empty_set(enc, Wd, bd, cnt, label):
    """Verify + document what enc(empty) decodes to under the (frozen) decoder: |z_empty| and the
    Spearman of the empty-set scores against the decoder bias and against train popularity."""
    from scipy.stats import spearmanr
    enc.eval()
    with torch.no_grad():
        ids = torch.zeros((1, 1), dtype=torch.long); vals = torch.zeros((1, 1))
        pad = torch.ones((1, 1), dtype=torch.bool); lvs = torch.zeros((1, 1), dtype=torch.long)
        z_empty = enc(ids, vals, pad, lvs)[0]
        sc = (z_empty @ Wd.T + bd).numpy()
    znorm = float(z_empty.norm())
    rho_bias = float(spearmanr(sc, bd.numpy()).statistic)
    rho_pop = float(spearmanr(sc, cnt).statistic) if cnt is not None else float("nan")
    log(f"[{label}] EMPTY-SET: |z|={znorm:.6f}; decode-vs-frozen-bias Spearman={rho_bias:.4f}; "
        f"vs train-popularity Spearman={rho_pop:.4f} "
        f"({'EXACT intercept identity (z=0 -> decoder bias)' if znorm < 1e-6 else 'z0/head trained away from 0 (allowed; measured here)'})")
    return znorm, rho_bias


def count_params(enc, decoder):
    tr_p = sum(p.numel() for p in enc.parameters() if p.requires_grad) \
         + sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    fr_p = sum(p.numel() for p in enc.parameters() if not p.requires_grad) \
         + sum(p.numel() for p in decoder.parameters() if not p.requires_grad)
    return tr_p, fr_p


# =============================================================================================
# T3' EDLAE DISTILLATION HOOK (score-CE; precompute top-k targets ONCE; default OFF)
# =============================================================================================
def precompute_teacher(users, teacher_npy, ni, topk=1000, temp=2.0, batch=512):
    """The known 50x optimisation: compute the EDLAE teacher distribution ONCE per user (from the user's
    FULL binary LIKE profile), store top-k (idx, prob). Returns (idx int32 [N,topk], prob float32 [N,topk]).
    Teacher logits = x_like @ B; top-k truncated temperature-softmax (Hinton KD; distill_edlae convention)."""
    if not os.path.exists(teacher_npy):
        raise SystemExit(f"[kd] teacher B npy not found: {teacher_npy}. Produce it: "
                         f"python src/baselines/edlae.py --p <best> --export_B <path>.npy")
    B = np.asarray(np.load(teacher_npy), dtype=np.float32)
    assert B.shape == (ni, ni), f"teacher B {B.shape} != ({ni},{ni})"
    N = len(users); kk = min(topk, ni)
    t_idx = np.zeros((N, kk), np.int32); t_prob = np.zeros((N, kk), np.float32)
    log(f"[kd] precomputing EDLAE teacher top-{kk} targets for {N} users (once)")
    for st in range(0, N, batch):
        rows = list(range(st, min(st + batch, N)))
        x = np.zeros((len(rows), ni), np.float32)
        for r, u in enumerate(rows):
            x[r, users[u]["liked"]] = 1.0
        logits = x @ B
        idx = np.argpartition(-logits, kk - 1, axis=1)[:, :kk]
        rr = np.arange(len(rows))[:, None]
        top = logits[rr, idx] / float(temp)
        top = top - top.max(axis=1, keepdims=True)
        ex = np.exp(top); ex /= ex.sum(axis=1, keepdims=True)
        t_idx[st:st + len(rows)] = idx.astype(np.int32)
        t_prob[st:st + len(rows)] = ex.astype(np.float32)
    return t_idx, t_prob


def kd_ce(logsm, uidx, t_idx, t_prob):
    """KD cross-entropy on the precomputed top-k teacher, PER EXAMPLE: -sum_k prob_k * logsm[idx_k]."""
    ti = torch.from_numpy(t_idx[uidx].astype(np.int64))          # (B,k)
    tp = torch.from_numpy(t_prob[uidx])                          # (B,k)
    picked = torch.gather(logsm, 1, ti)                          # (B,k)
    return -(tp * picked).sum(-1)                                # (B,)


# ---- repair 1: lambda_z schedule (subset-size ramp x epoch anneal) ----
RAMP_LO, RAMP_HI = 8, 30           # recvae mode: ramp(k)=0 for k<=8, 1 for k>=30
FULL_KD_LO, FULL_KD_HI = 30, 60    # --full_kd plan-B: latent term ONLY at large subsets


def lam_ramp(k, lo=RAMP_LO, hi=RAMP_HI):
    """ramp(k): 0 for k<=lo, linear to 1 at k>=hi."""
    return float(np.clip((k - lo) / (hi - lo), 0.0, 1.0))


def lam_anneal(ep, lambda_z, anneal_epochs):
    """Global anneal high->0: latent KD is a full-profile WARM-START; NLL owns the endgame."""
    return lambda_z * max(0.0, 1.0 - ep / max(anneal_epochs, 1))


# =============================================================================================
# MODEL BUILD  (teacher-mode wiring shared by train / dry_run / smoke)
# =============================================================================================
def level_valence(levels=None):
    """Centered valence v(level) in [-1,+1], NEUTRAL at the 3-star point (level 5), linear in stars,
    clipped: v = clip((star - 3.0)/2.0, -1, 1). Levels 0..9 <-> stars 0.5..5.0."""
    lv = np.arange(NLEV) if levels is None else np.asarray(levels)
    star = (lv + 1.0) / 2.0
    return np.clip((star - 3.0) / 2.0, -1.0, 1.0)


# SIGN-PRIOR v2 (archaeology verdict 2026-07-23): piecewise gamma init. ALL like levels (>=7, i.e.
# rating > 3.5) start at FULL +1.0 (pb2 parity -- v1's graded positives halved the dominant like signal);
# the 3/3.5-star neutral band starts weakly positive (+0.25); dislike levels stay GRADED NEGATIVE
# (hard-wired negation kept: vision guard / G3). Levels 0..9 = stars 0.5..5.0.
SIGN_GAMMA_V2 = [-1.0, -1.0, -1.0, -0.75, -0.5,   # 0.5-2.5 stars: graded negation (descending)
                 0.25, 0.25,                       # 3.0-3.5 stars: neutral band, weakly positive
                 1.0, 1.0, 1.0]                    # 4.0-5.0 stars: FULL positive (pb2 parity)


def apply_sign_prior(enc):
    """SIGN INIT ON GAMMA v2 (repair 2 + archaeology verdict; insurance vs the RUNG1
    sign-unreachable-from-null-init failure, memory `token-fusion-signed-values`):
    gamma(level) = SIGN_GAMMA_V2[level] broadcast across dims -- a hated token STARTS as a scaled -e_i
    (PER-ITEM negation, not a global direction), every LIKED token at full +e_i; beta ZERO-init.
    The earlier beta-along-u-bar variant is REMOVED (popularity-axis seeding); the v1 linear-valence
    gamma is REPLACED (it halved the positive signal on 4/4.5-star tokens)."""
    with torch.no_grad():
        v = torch.tensor(SIGN_GAMMA_V2, dtype=torch.float32)              # (NLEV,)
        enc.gamma.weight.copy_(v.unsqueeze(1).expand(-1, enc.d).contiguous())
        nn.init.zeros_(enc.beta.weight)
    log(f"[model] SIGN INIT v2 applied: gamma(level)={SIGN_GAMMA_V2} "
        f"(likes full +1, neutral +0.25, dislikes graded negative); beta=0")



def build_model(args, ni, cnt, teacher_override=None):
    """Returns (enc, decoder, teacher, trainable_params, groups). teacher_override: smoke fake."""
    if args.arch == "i25":
        # RESIDUAL fold (2026-07-23): frozen native RecVAE encoder inside the forward; loss teacher OFF.
        src = teacher_override if teacher_override is not None else \
            load_recvae_teacher(ni, hidden=args.t_hidden, latent=args.t_latent)
        n_conc = int(getattr(args, "n_concepts", 0)) if getattr(args, "concept_tokens", False) else 0
        enc = I25Encoder(ni, src, d_lat=args.t_latent, token_mode=args.token, n_concepts=n_conc)
        if n_conc > 0:
            log(f"[model] ARM C-FULL: concept-token channel ON ({n_conc} concepts x {args.t_latent} = "
                f"{n_conc * args.t_latent:,} extra TRAINABLE params; whitened init applied by train())")
        decoder = nn.Linear(args.t_latent, ni)
        with torch.no_grad():
            decoder.weight.copy_(src.decoder.weight)
            decoder.bias.copy_(src.decoder.bias)
            norms = src.decoder.weight.norm(dim=1).clamp_min(1e-8)
            enc.item_emb.weight.copy_(src.decoder.weight / norms.unsqueeze(1))   # normalized identities
        enc.item_emb.weight.requires_grad_(False)            # identities frozen in i25
        if args.train_decoder:                               # arm B (fallback): trainable at slow LR
            log("[model] i25 arm B: decoder TRAINABLE at slow LR (--train_decoder)")
        else:                                                # arm A (primary): frozen -> G0-identity
            decoder.weight.requires_grad_(False); decoder.bias.requires_grad_(False)
            log("[model] i25 arm A: decoder FROZEN RecVAE W+b (G0-identity restored)")
        teacher = None                                       # native encoder is in the FORWARD, not loss
        if getattr(args, "sign_prior", True):
            apply_sign_prior(enc)
        else:
            log("[model] sign_prior OFF (ablation arm)")
        params = [p for p in list(enc.parameters()) + list(decoder.parameters()) if p.requires_grad]
        if args.train_decoder:
            slow_ids = {id(decoder.weight), id(decoder.bias)}
            fast = [p for p in params if id(p) not in slow_ids]
            slow = [p for p in params if id(p) in slow_ids]
            groups = [{"params": fast}, {"params": slow, "lr": args.lr * args.warm_lr_scale}]
        else:
            groups = [{"params": params}]
        tr_p, fr_p = count_params(enc, decoder)
        log(f"[model] arch=i25 d_lat={args.t_latent} TRAINABLE={tr_p:,} FROZEN={fr_p:,} "
            f"(+ unregistered frozen native encoder ~13.4M params, not counted/saved) "
            f"train_decoder={args.train_decoder}")
        return enc, decoder, teacher, params, groups
    if args.teacher == "warm_init":
        # pre-registered escalation (probe FAIL 2026-07-22): TRAINABLE decoder + identities, RecVAE ckpt
        # is INIT-ONLY -- the model is dropped from the loss path (returned teacher=None -> no latent KD).
        src = teacher_override if teacher_override is not None else \
            load_recvae_teacher(ni, hidden=args.t_hidden, latent=args.t_latent)
        d_out = args.t_latent
        enc = SetEncoder(ni, d=D, d_out=d_out, d_emb=d_out, token_mode=args.token, norm_feat=False)
        decoder = nn.Linear(d_out, ni)
        with torch.no_grad():
            decoder.weight.copy_(src.decoder.weight)         # TRAINABLE, warm init
            decoder.bias.copy_(src.decoder.bias)             # intercept starts at RecVAE's marginal
            norms = src.decoder.weight.norm(dim=1).clamp_min(1e-8)
            enc.item_emb.weight.copy_(src.decoder.weight / norms.unsqueeze(1))  # normalized ONCE at init
        enc.z0.requires_grad_(False)                         # gate + z0=0 -> enc(empty)=decoder bias(t)
        # NOTE --unfreeze_emb is a NO-OP in warm mode: identities are ALWAYS trainable here (and always
        # routed to the SLOW LR group below, since they were warm-started).
        if getattr(args, "full_kd", False):                  # plan-B: dense full-profile latent anchor
            teacher = src                                    # FROZEN, loss-only; decoder stays TRAINABLE
            log(f"[model] warm_init + full_kd: latent term at large subsets only "
                f"(ramp {FULL_KD_LO}->{FULL_KD_HI}, constant w={args.full_kd_w}, no anneal). "
                f"CAVEAT: if the trainable decoder drifts far from the ckpt geometry the KD target can "
                f"fight it -- mitigated by the slow decoder LR (warm_lr_scale={args.warm_lr_scale}).")
        else:
            teacher = None                                   # latent KD OFF; ckpt not in the loss path
        log("[model] warm_init: decoder W+b + item identities TRAINABLE (RecVAE init)")
    elif args.teacher == "recvae":
        teacher = teacher_override if teacher_override is not None else \
            load_recvae_teacher(ni, hidden=args.t_hidden, latent=args.t_latent)
        d_out = args.t_latent
        enc = SetEncoder(ni, d=D, d_out=d_out, d_emb=d_out, token_mode=args.token, norm_feat=True)
        decoder = nn.Linear(d_out, ni)
        with torch.no_grad():                                # FROZEN RecVAE decoder + bias, reused
            decoder.weight.copy_(teacher.decoder.weight)     # NOTE: bias = RecVAE's LEARNED marginal
            decoder.bias.copy_(teacher.decoder.bias)         # (NOT log-count popularity; repair 5)
            # repair 3: input token identity = NORMALIZED decoder rows; ||Wd[i]|| as scalar feature
            norms = teacher.decoder.weight.norm(dim=1).clamp_min(1e-8)
            enc.item_emb.weight.copy_(teacher.decoder.weight / norms.unsqueeze(1))
            enc.emb_norm.copy_(norms)
        decoder.weight.requires_grad_(False); decoder.bias.requires_grad_(False)
        enc.z0.requires_grad_(False)                         # repair 5: z0 FROZEN at 0 -> exact intercept
        if not args.unfreeze_emb:
            enc.item_emb.weight.requires_grad_(False)
    else:                                                    # --teacher none: from-scratch fallback arm
        teacher = None
        enc = SetEncoder(ni, d=D, d_out=D, token_mode=args.token)
        decoder = nn.Linear(D, ni)
        with torch.no_grad():                                # pop-prior bias (z0=0 -> cold pop)
            decoder.bias.copy_(torch.from_numpy(np.log(cnt / cnt.sum() + 1e-9).astype(np.float32)))
            nn.init.normal_(decoder.weight, std=0.02)
    if getattr(args, "sign_prior", True):
        apply_sign_prior(enc)                                # repair 2: signed gamma init (both modes)
    else:
        log("[model] sign_prior OFF (ablation arm): gamma=1, beta random (additive-null init)")
    # repair 6: frozen tensors EXCLUDED from the optimizer entirely (AdamW decay mutates at zero grad)
    params = [p for p in list(enc.parameters()) + list(decoder.parameters()) if p.requires_grad]
    # stabilization patch (2026-07-22 warm_init ep5 collapse): warm-started decoder+identities get a
    # REDUCED learning rate (lr * warm_lr_scale) -- training 7.4M warm params at the from-scratch
    # attention LR churned the good init (train NLL rose 6.64->7.79 = divergence).
    if args.teacher == "warm_init":
        slow_ids = {id(decoder.weight), id(decoder.bias), id(enc.item_emb.weight)}
        slow = [p for p in params if id(p) in slow_ids]
        fast = [p for p in params if id(p) not in slow_ids]
        groups = [{"params": fast},
                  {"params": slow, "lr": args.lr * args.warm_lr_scale}]
        log(f"[model] param groups: fast={sum(p.numel() for p in fast):,}@lr={args.lr} "
            f"slow(warm decoder+identities)={sum(p.numel() for p in slow):,}"
            f"@lr={args.lr * args.warm_lr_scale} (warm_lr_scale={args.warm_lr_scale})")
    else:
        groups = [{"params": params}]
    tr_p, fr_p = count_params(enc, decoder)
    log(f"[model] teacher={args.teacher} d_int={D} d_out={enc.d_out} "
        f"TRAINABLE={tr_p:,} FROZEN={fr_p:,} (unfreeze_emb={args.unfreeze_emb})")
    return enc, decoder, teacher, params, groups


def nll_guard_step(guard, ep_nll, opt, enc, decoder, ckb):
    """Plateau-rescue (patch 4): if epoch train NLL rises >10% over its running min for 2 CONSECUTIVE
    epochs, halve ALL group LRs and reload the best checkpoint weights. guard = {'min','streak'} dict,
    mutated in place. Returns True when a rescue fired."""
    if ep_nll < guard["min"]:
        guard["min"] = ep_nll; guard["streak"] = 0
        return False
    if ep_nll > 1.10 * guard["min"]:
        guard["streak"] += 1
        if guard["streak"] >= 2:
            for g in opt.param_groups:
                g["lr"] *= 0.5
            if ckb and os.path.exists(ckb):
                blob = torch.load(ckb, map_location="cpu")
                enc.load_state_dict(blob["enc"]); decoder.load_state_dict(blob["decoder"])
                reloaded = "best ckpt reloaded"
            else:
                reloaded = "no best ckpt yet (weights kept)"
            guard["streak"] = 0
            log(f"[guard] train-NLL divergence (> {1.10 * guard['min']:.4f} twice): "
                f"ALL LRs halved -> {[round(g['lr'], 6) for g in opt.param_groups]}; {reloaded}; "
                f"continuing at halved LRs")
            return True
    else:
        guard["streak"] = 0
    return False


def early_stop_step(bad, improved, rescued, patience):
    """Early-stop counter update (2026-07-23 fix: 'done after rescue' bug). A guard rescue RESETS the
    no-improvement counter -- the reloaded-best model at halved LRs deserves fresh patience. Without
    this, a rescue firing when bad == patience-1 still incremented bad past patience the SAME epoch
    (the diverged epoch's val can't beat best by construction) and the loop exited right after the
    rescue. Returns (bad, stop)."""
    if improved or rescued:
        return 0, False
    bad += 1
    return bad, bad >= patience


def batch_loss(enc, decoder, teacher, exs, ni, args, lam_glob=None, t_idx=None, t_prob=None):
    """Shared loss for one packed batch (v3 per-example schedule, repairs 1+4).
    exs: list of (user_idx, (inp, lv, sv, tgt, negs)). lam_glob = epoch-annealed global lambda_z.
    loss = mean_i [ lam_i * zMSE_i + (1-lam_i) * rank_i ],   lam_i = lam_glob * ramp(k_i)
    rank_i = like-NLL_i + w_neg * mean-logprob(held dislikes_i)  [+ alpha_kd EDLAE CE mix].
    Returns (loss, nll_float, zmse_float)."""
    B = len(exs)
    packrows = [(e[0], e[1], e[2]) for _, e in exs]
    ids, vals, pad, lvs = pack_tokens(packrows, binarize=args.ablate_binarized)
    tgt = torch.zeros((B, ni), dtype=torch.float32)
    neg = torch.zeros((B, ni), dtype=torch.float32)
    for r, (_, e) in enumerate(exs):
        tgt[r, e[3]] = 1.0
        if len(e) > 4 and len(e[4]) > 0:
            neg[r, e[4]] = 1.0
    z = enc(ids, vals, pad, lvs)
    logits = z @ decoder.weight.T + decoder.bias
    logsm = F.log_softmax(logits, dim=-1)
    nll_vec = -((logsm * tgt).sum(-1) / tgt.sum(-1).clamp_min(1.0))               # (B,)
    # repair 4 + BOUNDEDNESS FIX (2026-07-23, ep4-7 divergence root cause CONFIRMED): held dislikes as
    # down-weighted explicit negatives, CLAMPED. The unclamped form w_neg * mean(log p(dislike)) is
    # UNBOUNDED BELOW -- once the softmax sharpens (~ep3-4) the optimizer reduces total loss without
    # limit by driving disliked logits to -inf, dragging the like-NLL up (the observed signature in all
    # three warm runs). Clamp: max(log p, FLOOR), FLOOR = log(1/n_items) - NEG_FLOOR_OFFSET -- once a
    # dislike is ranked ~e^2 below uniform it stops generating gradient.
    floor = -(np.log(ni) + NEG_FLOOR_OFFSET)
    neg_vec = (torch.clamp(logsm, min=floor) * neg).sum(-1) / neg.sum(-1).clamp_min(1.0)   # bounded
    rank_vec = nll_vec + args.w_neg * neg_vec
    with torch.no_grad():                                    # diagnostic: fraction of negs at the floor
        n_negs = float(neg.sum())
        floor_frac = (float(((logsm <= floor) & neg.bool()).sum()) / n_negs) if n_negs > 0 \
            else float("nan")
    if args.alpha_kd > 0:
        uidx = np.array([i for i, _ in exs], np.int64)
        rank_vec = (1.0 - args.alpha_kd) * rank_vec + args.alpha_kd * kd_ce(logsm, uidx, t_idx, t_prob)
    zmse_mean = float("nan")
    if teacher is not None and lam_glob is not None and lam_glob > 0:
        # review decision 2026-07-22: teacher input = LIKES-ONLY (level >= LIKE_MIN_LEVEL, i.e. > 3.5
        # stars) binarized -- RecVAE never saw dislikes as positives; a hated film fed as binary 1 would
        # make z_T believe the user likes it. Dislikes stay in the STUDENT tokens + the w_neg term.
        liked_in = [e[0][e[1] >= LIKE_MIN_LEVEL] for _, e in exs]
        lo, hi = (FULL_KD_LO, FULL_KD_HI) if getattr(args, "full_kd", False) else (RAMP_LO, RAMP_HI)
        ramp = torch.tensor([lam_ramp(len(e[0]), lo, hi) if len(lk) > 0 else 0.0
                             for (_, e), lk in zip(exs, liked_in)], dtype=torch.float32)   # (B,)
        lam = lam_glob * ramp
        sel = (ramp > 0).nonzero(as_tuple=True)[0]
        mse_vec = torch.zeros(B)
        if sel.numel() > 0:                                  # teacher forward ONLY for ramp>0 rows
            z_T = teacher_latent(teacher, [liked_in[int(r)] for r in sel], ni)
            mse_sel = ((z[sel] - z_T) ** 2).mean(-1)
            mse_vec = mse_vec.index_copy(0, sel, mse_sel)
            zmse_mean = float(mse_sel.mean())
        loss = (lam * mse_vec + (1.0 - lam) * rank_vec).mean()
        return loss, float(nll_vec.mean()), zmse_mean, floor_frac
    return rank_vec.mean(), float(nll_vec.mean()), zmse_mean, floor_frac


def cold_concepts_full10(enc, Wd, bd, sel_val, m, va_tr, va_te, ni, batch=500):
    """Concepts-ONLY cold val full@10 (ARM C-FULL selection axis): fold m concept tokens (ids ni+c at
    sel_value_to_level(v)) with NO items; mask = full canonical fold-in (cold parity); users with no
    SEL concept excluded."""
    n = va_tr.shape[0]
    accs = []
    enc.eval()
    with torch.no_grad():
        for st in range(0, n, batch):
            rows = list(range(st, min(st + batch, n)))
            packrows = []
            for r in rows:
                if sel_val[r] is None:
                    packrows.append((np.empty(0, np.int64), np.empty(0, np.int64),
                                     np.empty(0, np.float32)))
                else:
                    cs, lvls = sel_val[r]                    # PRE-MAPPED levels (signed-aware)
                    cid = (ni + cs[:m]).astype(np.int64)
                    clv = np.asarray(lvls[:m], np.int64)
                    packrows.append((cid, clv, np.zeros(len(cid), np.float32)))
            ids, vals, pad, lvs = pack_tokens(packrows)
            S = (enc(ids, vals, pad, lvs) @ Wd.T + bd).numpy().astype(np.float32)
            S[va_tr[rows].nonzero()] = -np.inf
            te = va_te[rows]
            keep = (np.asarray(te.getnnz(axis=1)).ravel() > 0) & \
                   np.array([sel_val[r] is not None for r in rows])
            if keep.any():
                accs.append(M.NDCG_binary_at_k_batch(S[keep], te[np.flatnonzero(keep)], k=10))
    return float(np.concatenate(accs).mean()) if accs else float("nan")


# =============================================================================================
# TRAIN
# =============================================================================================
def train(args):
    os.makedirs(CKPT_DIR, exist_ok=True); os.makedirs(OUTDIR, exist_ok=True)
    meta = M.load_meta(PROC); ni = meta["n_items"]
    train_mat = M.load_train(ni, PROC)
    head_mask, cnt = compute_head_mask(train_mat, ni)
    unique_uid, tr_set, vd_set, te_set, n_train, raw, show2id, usid = reproduce_partition()
    users = build_train_profiles(raw, tr_set, show2id,
                                 max_users=(args.max_users if args.max_users else None))

    # ---- ARM C-FULL concept setup (--concept_tokens; default OFF -> t2final provenance untouched) ----
    Mm = grate = member_sets = tags = None
    if args.concept_tokens:
        import pandas as pd
        from concept_fold import build_member_matrix
        gpath = os.path.join(_ROOT, "data", "movielens", "genome-scores.csv")
        m2s = {int(m): i for i, m in enumerate(usid)}
        gg = pd.read_csv(gpath); gg = gg[gg["relevance"] >= 0.5]; gg = gg[gg["movieId"].isin(m2s)]
        gg["sid"] = gg["movieId"].map(m2s).astype(np.int64)
        members = {int(t): np.sort(s["sid"].values) for t, s in gg.groupby("tagId")}
        members = {t: mm for t, mm in members.items() if len(mm) >= 30}
        tags = sorted(members.keys())
        Mm = build_member_matrix(members, tags, ni)
        grate = (cnt @ np.asarray(Mm.todense())) / max(cnt.sum(), 1e-9)
        member_sets = {i: members[t] for i, t in enumerate(tags)}
        args.n_concepts = len(tags)
        log(f"[cfull] concept channel: {len(tags)} genome concepts (>=30 members); "
            f"p_concept_ex={args.p_concept_ex} (item-only examples keep the item pathway fed)")
        signed_prereg = signed_item_mean = None
        if getattr(args, "signed_concepts", False):
            from signed_answers import compute_prereg
            signed_prereg, signed_item_mean = compute_prereg(raw, tr_set, show2id, ni, Mm, grate)
            log(f"[cfull] SIGNED concepts: w_val={signed_prereg['w_val']:.3f} "
                f"t_like={signed_prereg['t_like_p60pos']:.3f} "
                f"t_neg={signed_prereg['t_neg_absp25']:.3f} m<=16, dislike levels 0..4")

    enc, decoder, teacher, params, groups = build_model(args, ni, cnt)
    if args.concept_tokens:
        # whitened member-centroid init for the TRAINABLE concept identities (the July geometry prior)
        from belief_layer import build_concept_dirs           # lazy (belief_layer imports this module)
        members_by_tag = {t: member_sets[i] for i, t in enumerate(tags)}
        d_c, _, _ = build_concept_dirs(decoder.weight.detach(), members_by_tag, tags)
        with torch.no_grad():
            enc.concept_emb.weight.copy_(d_c)
        assert enc.concept_emb.weight.requires_grad, "concept_emb must be TRAINABLE (Arm C-full)"
        assert any(p is enc.concept_emb.weight for p in params), "concept_emb missing from optimizer"
        log(f"[cfull] concept_emb init = whitened member centroids ({len(tags)}x{enc.d})")
    report_empty_set(enc, decoder.weight.detach(), decoder.bias.detach(), cnt, "init")
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=1e-4)

    t_idx = t_prob = None
    if args.alpha_kd > 0:
        t_idx, t_prob = precompute_teacher(users, args.teacher_npy, ni, topk=args.kd_topk, temp=args.kd_temp)

    lens = np.array([len(u["items"]) for u in users])
    batches_all = make_batches(users, np.argsort(lens))
    log(f"[train] {len(users)} users; profiles min={lens.min()} med={int(np.median(lens))} max={lens.max()}; "
        f"{len(batches_all)} adaptive batches; token={args.token} teacher={args.teacher} "
        f"lambda_z={args.lambda_z} alpha_kd={args.alpha_kd} ablate_binarized={args.ablate_binarized}")

    ck = os.path.join(CKPT_DIR, f"{args.tag}.pt"); ckb = os.path.join(CKPT_DIR, f"{args.tag}_best.pt")
    start_ep, best, bad = 0, -1.0, 0
    guard = {"min": float("inf"), "streak": 0}
    if args.resume_from_best and os.path.exists(ckb):
        # patch 3: restart from the BEST checkpoint's weights (e.g. post-collapse). The best ckpt
        # stores no optimizer state -> Adam moments start FRESH at the configured (grouped) LRs.
        blob = torch.load(ckb, map_location="cpu")
        enc.load_state_dict(blob["enc"]); decoder.load_state_dict(blob["decoder"])
        start_ep = blob["epoch"]; best = blob.get("val_full", -1.0)
        log(f"[train] RESUMED FROM BEST ep{start_ep} (val_full={best:.4f}); fresh optimizer, "
            f"group LRs={[round(g['lr'], 6) for g in opt.param_groups]}")
    elif args.resume and os.path.exists(ck):
        blob = torch.load(ck, map_location="cpu")
        enc.load_state_dict(blob["enc"]); decoder.load_state_dict(blob["decoder"])
        opt.load_state_dict(blob["opt"]); start_ep = blob["epoch"]; best = blob.get("best", -1.0)
        bad = blob.get("bad", 0)
        guard = blob.get("guard", guard)
        log(f"[train] RESUMED ep{start_ep} best={best:.4f} "
            f"LRs={[round(g['lr'], 6) for g in opt.param_groups]}")

    W0 = decoder.weight.detach().clone(); b0 = decoder.bias.detach().clone()   # repair 6 drift ref
    E0 = enc.item_emb.weight.detach().clone()
    # prebuilt val artifacts (ONCE): full graded fold-in + the FIXED cold k-subsets (COLD_SEED --
    # same subsets every epoch, so the cold curve is comparable across epochs)
    L_val, _ = build_graded_eval_matrix(raw, unique_uid, show2id, usid, "validation")
    va_tr, va_te = M.load_val(ni, PROC)
    Lk2 = Lk8 = None
    if not args.no_cold_val:
        Lk2 = truncate_graded(L_val, 2, COLD_SEED)
        Lk8 = truncate_graded(L_val, 8, COLD_SEED + 1)
        log(f"[train] cold-val subsets built once: k=2 nnz={Lk2.nnz}, k=8 nnz={Lk8.nnz} (seed {COLD_SEED})")
    sel_val = None
    if args.concept_tokens and not args.no_cold_val:
        if getattr(args, "signed_concepts", False):
            # SIGNED monitor: top-8 by |v|, C_NEG cap, levels via value_to_level (incl. dislikes)
            from signed_answers import signed_values, cap_negatives, value_to_level, BAND_REFUSE
            counts_v = None
            items_l = []; stars_l = []
            for r in range(va_tr.shape[0]):
                s_, e_ = L_val.indptr[r], L_val.indptr[r + 1]
                sids = L_val.indices[s_:e_].astype(np.int64)
                lvls = (L_val.data[s_:e_] - 1.0)
                items_l.append(sids); stars_l.append(((lvls + 1.0) / 2.0).astype(np.float32))
            Vv, Fv, Bv, ansv = signed_values(items_l, stars_l, signed_item_mean, Mm, grate, ni,
                                             signed_prereg, apply_neg_cap=False)
            sel_val = []
            for r in range(va_tr.shape[0]):
                cand = np.flatnonzero(ansv[r] & (Bv[r] != BAND_REFUSE))
                if len(cand) == 0:
                    sel_val.append(None); continue
                o = cand[np.argsort(-np.abs(Vv[r, cand]))][:8]
                vv = cap_negatives([float(Vv[r, c]) for c in o])
                sel_val.append((o.astype(np.int64),
                                np.asarray([value_to_level(v) for v in vv], np.int64)))
        else:
            # per-val-user top-8 SEL concepts (LEGACY clip path, kept for unsigned runs)
            counts_v = np.asarray((va_tr @ Mm).todense())
            nu_v = np.asarray(va_tr.sum(axis=1)).ravel().clip(min=1)
            lift_v = (counts_v / nu_v[:, None]) / np.maximum(grate[None, :], 1e-12)
            lift_v[counts_v < 2] = -np.inf
            sel_val = []
            for r in range(va_tr.shape[0]):
                pos = np.flatnonzero(np.isfinite(lift_v[r]) & (lift_v[r] > 1.0))
                if len(pos) == 0:
                    sel_val.append(None); continue
                o = pos[np.argsort(-lift_v[r][pos])][:8]
                l1 = np.log(lift_v[r][o[0]])
                vv = np.clip(np.log(lift_v[r][o]) / max(l1, 1e-9), 0.25, 1.0)
                sel_val.append((o.astype(np.int64),
                                np.asarray([sel_value_to_level(v) for v in vv], np.int64)))
        log(f"[cfull] val concepts-only sets built once: "
            f"{sum(1 for s in sel_val if s is not None)} users with >=1 concept "
            f"({'SIGNED' if getattr(args, 'signed_concepts', False) else 'legacy clip'})")

    # S2 NATIVE-SEEDED BEST (v4 review): evaluate the INIT model (delta==0) on val and seed `best` +
    # the best checkpoint with it BEFORE epoch 1 -- a below-native G0 outcome becomes impossible
    # (early stop can never keep a checkpoint worse than the anchored init).
    native_ref_f10 = G0_BAR          # fallback reference for the cold-primary tie constraint
    best_cold = -1.0                 # cold-composite incumbent for --select_cold
    if start_ep == 0 and best < 0:
        vm0 = M.evaluate(make_graded_predict_fn(enc, decoder.weight.detach(), decoder.bias.detach(),
                                                L_val, binarize=args.ablate_binarized),
                         va_tr, va_te, batch_size=500, head_mask=head_mask)
        best = vm0["ndcg@10"]
        native_ref_f10 = best        # the anchored-init full@10 IS the tie reference on this val cohort
        torch.save({"enc": enc.state_dict(), "decoder": decoder.state_dict(),
                    "epoch": 0, "val_full": best, "val_tail": vm0["tail_ndcg@10"]}, ckb)
        log(f"[init] VAL full@10={best:.4f} tail@10={vm0['tail_ndcg@10']:.4f} "
            f"(native{'-anchored' if args.arch == 'i25' else ''} init; best SEEDED -> "
            f"below-native G0 impossible)")

    p0, p1 = parse_p_interview(args.p_interview)
    for ep in range(start_ep, args.epochs):
        enc.train(); rng = np.random.default_rng(ep)
        # lam selection (2026-07-23 fix): warm mode without full_kd PINS lam=0 -- the rank loss runs at
        # CONSTANT weight 1.0. (NOTE for the record: with teacher=None batch_loss already returned the
        # pure rank loss at weight 1.0; the annealing lam_glob was computed and LOGGED but never entered
        # the loss. The pin makes the log truthful and forecloses the creep path permanently.)
        if args.arch == "i25" or (args.teacher == "warm_init" and not args.full_kd):
            lam_glob = 0.0                   # i25: native encoder is in the FORWARD, never a loss target
        elif args.full_kd:
            lam_glob = args.full_kd_w        # constant convex weight, no anneal (as built)
        else:
            lam_glob = lam_anneal(ep, args.lambda_z, args.anneal_epochs)   # repair 1 anneal (recvae mode)
        p_int = p_interview_at(ep, p0, p1, args.anneal_epochs)                 # curriculum schedule
        order = list(range(len(batches_all))); rng.shuffle(order)
        t0 = time.time(); run_n = 0.0; run_z = 0.0; nb = 0; run_f = 0.0; nf = 0
        for bi in order:
            bat = batches_all[bi]
            if args.concept_tokens:
                # C-full mix: concept-curriculum examples w.p. p_concept_ex, else plain item examples
                # (the item pathway must never starve -- author spec)
                if getattr(args, "signed_concepts", False):
                    exs = [(i, make_concept_example_signed(users[i], rng, Mm, grate, member_sets,
                                                           ni, signed_item_mean, signed_prereg)
                            if rng.random() < args.p_concept_ex
                            else make_input_target(users[i], rng, p_int=p_int)) for i in bat]
                else:
                    exs = [(i, make_concept_example(users[i], rng, Mm, grate, member_sets, ni,
                                                    p_conc_only=0.35)
                            if rng.random() < args.p_concept_ex
                            else make_input_target(users[i], rng, p_int=p_int)) for i in bat]
            else:
                exs = [(i, make_input_target(users[i], rng, p_int=p_int)) for i in bat]
            exs = [(i, e) for i, e in exs if e is not None]
            if not exs:
                continue
            loss, nll_v, mse_v, ff_v = batch_loss(enc, decoder, teacher, exs, ni, args, lam_glob, t_idx, t_prob)
            opt.zero_grad(); loss.backward()
            if args.clip > 0:                                # patch 2: global grad-norm clipping
                torch.nn.utils.clip_grad_norm_(params, args.clip)
            opt.step()
            run_n += nll_v; run_z += (0.0 if np.isnan(mse_v) else mse_v); nb += 1
            if not np.isnan(ff_v):
                run_f += ff_v; nf += 1
            if nb % 100 == 0:
                log(f"  ep{ep} b{nb}/{len(order)} NLL={run_n/nb:.4f} zMSE={run_z/nb:.4f} "
                    f"negfloor={run_f/max(nf,1):.2f} {(time.time()-t0)/60:.1f}m")
        if args.arch == "i25":                                                 # repair 6: trunk drift
            assert torch.equal(enc.item_emb.weight, E0), "FROZEN i25 identities drifted"
            if not args.train_decoder:
                assert torch.equal(decoder.weight, W0) and torch.equal(decoder.bias, b0), \
                    "FROZEN i25 arm-A decoder drifted"
        elif args.teacher == "recvae":
            assert torch.equal(decoder.weight, W0) and torch.equal(decoder.bias, b0), \
                "FROZEN decoder drifted (||dWd|| != 0)"
            assert args.unfreeze_emb or torch.equal(enc.item_emb.weight, E0), \
                "FROZEN item identities drifted"
        Wd_, bd_ = decoder.weight.detach(), decoder.bias.detach()
        predict = make_graded_predict_fn(enc, Wd_, bd_, L_val, binarize=args.ablate_binarized)
        vm = M.evaluate(predict, va_tr, va_te, batch_size=500, head_mask=head_mask)
        f10, t10 = vm["ndcg@10"], vm["tail_ndcg@10"]
        cold_str = ""
        cm2 = cm8 = None
        if not args.no_cold_val:                             # cold val: k-subset fold, FULL candidate mask
            c2 = cold_full10(enc, Wd_, bd_, Lk2, va_tr, va_te, head_mask, args.ablate_binarized)
            c8 = cold_full10(enc, Wd_, bd_, Lk8, va_tr, va_te, head_mask, args.ablate_binarized)
            cold_str = f"coldk2={c2:.4f} coldk8={c8:.4f} "
            if args.concept_tokens and sel_val is not None:  # C-full: concepts-only selection axis
                cm2 = cold_concepts_full10(enc, Wd_, bd_, sel_val, 2, va_tr, va_te, ni)
                cm8 = cold_concepts_full10(enc, Wd_, bd_, sel_val, 8, va_tr, va_te, ni)
                cold_str += f"concm2={cm2:.4f} concm8={cm8:.4f} "
        log(f"[ep{ep+1}] NLL={run_n/max(nb,1):.4f} zMSE={run_z/max(nb,1):.4f} lam_glob={lam_glob:.3f} "
            f"p_int={p_int:.2f} negfloor={run_f/max(nf,1):.2f} "
            f"VAL full@10={f10:.4f} tail@10={t10:.4f} {cold_str}ndcg@100={vm['ndcg@100']:.4f} "
            f"vs bar {G0_BAR:.4f} ({f10 - G0_BAR:+.4f}) ({(time.time()-t0)/60:.1f}m)")
        rescued = bool(args.nll_guard) and \
            nll_guard_step(guard, run_n / max(nb, 1), opt, enc, decoder, ckb)   # patch 4: rescue
        stop = False
        # SELECTION RULE (2026-07-24, author-approved): cold-primary SUBJECT TO full-profile tie.
        # An epoch is best if its COLD composite (mean of coldk2,coldk8) beats the incumbent AND its
        # full@10 stays within G0_TIE_CI of the seeded native reference. Falls back to full@10-primary
        # when cold-val is off. Patience runs on this selection criterion.
        # C-full extension (author 2026-07-24): the composite includes the concepts-only m2/m8 axis
        if not args.no_cold_val:
            cold_comp = (np.mean([c2, c8, cm2, cm8]) if (args.concept_tokens and cm2 is not None)
                         else (c2 + c8) / 2.0)
        else:
            cold_comp = None
        if args.select_cold and cold_comp is not None:
            full_ok = f10 >= native_ref_f10 - G0_TIE_CI
            improved = full_ok and cold_comp > best_cold
            if improved:
                best_cold = cold_comp; best = max(best, f10); bad = 0
                torch.save({"enc": enc.state_dict(), "decoder": decoder.state_dict(),
                            "epoch": ep + 1, "val_full": f10, "val_tail": t10,
                            "coldk2": c2, "coldk8": c8}, ckb)
                log(f"[train] new best (cold-primary): cold={cold_comp:.4f} full={f10:.4f} "
                    f"(full_ok={full_ok})")
            else:
                bad, stop = early_stop_step(bad, False, rescued, args.patience)
                log(f"[train] no cold improvement over {best_cold:.4f} (full_ok={full_ok}) "
                    f"({bad}/{args.patience})"
                    + (" [guard rescue: patience counter reset, continuing]" if rescued else ""))
        elif f10 > best:
            best = f10; bad = 0
            torch.save({"enc": enc.state_dict(), "decoder": decoder.state_dict(),
                        "epoch": ep + 1, "val_full": f10, "val_tail": t10}, ckb)
        else:
            bad, stop = early_stop_step(bad, False, rescued, args.patience)
            log(f"[train] no improvement over {best:.4f} ({bad}/{args.patience})"
                + (" [guard rescue: patience counter reset, continuing]" if rescued else ""))
        # save the resume ckpt AFTER the best-update so 'best' (and 'bad') survive a resume
        # (ordering bug fix 2026-07-23: previously saved stale best -> RESUMED best=-1)
        torch.save({"enc": enc.state_dict(), "decoder": decoder.state_dict(), "opt": opt.state_dict(),
                    "epoch": ep + 1, "best": best, "bad": bad, "val_full": f10, "val_tail": t10,
                    "guard": guard}, ck)
        if stop:
            log(f"[train] CONVERGED (val flat {args.patience} epochs)"); break
    log(f"[train] done best val full@10={best:.4f}")

    # ---- test eval on the BEST checkpoint -> tower_t2.json ----
    if os.path.exists(ckb):
        blob = torch.load(ckb, map_location="cpu")
        enc.load_state_dict(blob["enc"]); decoder.load_state_dict(blob["decoder"])
    report_empty_set(enc, decoder.weight.detach(), decoder.bias.detach(), cnt, "final")
    tm = eval_split(enc, decoder.weight.detach(), decoder.bias.detach(), raw, unique_uid, show2id,
                    usid, "test", head_mask, binarize=args.ablate_binarized)
    tr_p, fr_p = count_params(enc, decoder)
    out = {"model": "tower_t2 (pb2-class graded-native set encoder, v3)", "token": args.token,
           "teacher": args.teacher, "lambda_z": args.lambda_z, "anneal_epochs": args.anneal_epochs,
           "w_neg": args.w_neg, "alpha_kd": args.alpha_kd, "sign_prior": args.sign_prior,
           "unfreeze_emb": args.unfreeze_emb, "ablate_binarized": args.ablate_binarized,
           "concept_tokens": bool(args.concept_tokens),
           "n_concepts": int(getattr(args, "n_concepts", 0)),
           "p_concept_ex": args.p_concept_ex if args.concept_tokens else None,
           "trainable_params": tr_p, "frozen_params": fr_p, "n_items": ni,
           "ndcg@10": tm["ndcg@10"], "ndcg@10_se": tm["ndcg@10_se"],
           "tail_ndcg@10": tm["tail_ndcg@10"], "ndcg@100": tm["ndcg@100"],
           "recall@20": tm["recall@20"], "recall@50": tm["recall@50"], "val_full@10_best": best,
           "geometry": {"warm_init": "TRAINABLE decoder+identities, RecVAE warm init, latent KD OFF",
                        "recvae": "FROZEN RecVAE decoder+bias+identities (recvae_ml25m_liang.pt best_state)",
                        "none": "from scratch; pop-init decoder bias"}[args.teacher]}
    if args.arch == "i25" or args.teacher in ("recvae", "warm_init"):   # repair 7: G0 reporting split
        # G0-strength: CI-tie of the tower's full-fold test NDCG@10 vs the frozen RecVAE's own
        # (in warm_init the tie TARGET stays the frozen ckpt RecVAE -- reload it, the trained decoder
        # has moved away from it by design)
        g0_ref = teacher if teacher is not None else \
            load_recvae_teacher(ni, hidden=args.t_hidden, latent=args.t_latent)
        te_tr, te_te = M.load_test(ni, PROC)
        rec_res = M.evaluate(R.make_predict_fn(g0_ref), te_tr, te_te, batch_size=500,
                             head_mask=head_mask)
        diff = tm["ndcg@10"] - rec_res["ndcg@10"]
        ci = 1.96 * float(np.sqrt(tm["ndcg@10_se"] ** 2 + rec_res["ndcg@10_se"] ** 2))
        out["G0_strength"] = {"tower_full@10": tm["ndcg@10"], "recvae_full@10": rec_res["ndcg@10"],
                              "recvae_tail@10": rec_res["tail_ndcg@10"], "diff": diff,
                              "ci95_halfwidth": ci, "tie": bool(abs(diff) <= ci),
                              "ge_native_minus_ci": bool(diff >= -ci)}
        if args.arch == "i25":
            out["G0_strength"]["criterion"] = (
                "i25 residual form: identity only at init (z=native+delta); post-training the "
                "val-selected residual makes the criterion tower >= native - CI (ge_native_minus_ci)")
        # G0-identity: mechanical bit-identity of the intercept + harness pass-through
        with torch.no_grad():
            z_e = enc(torch.zeros((1, 1), dtype=torch.long), torch.zeros((1, 1)),
                      torch.ones((1, 1), dtype=torch.bool), torch.zeros((1, 1), dtype=torch.long))
            sc_e = (z_e @ decoder.weight.T + decoder.bias)[0]
        out["G0_identity"] = {
            "empty_set_equals_frozen_bias": bool(torch.equal(sc_e, decoder.bias.detach())),
            "z_empty_norm": float(z_e.norm())}
        log(f"[G0] strength: tower {tm['ndcg@10']:.4f} vs recvae {rec_res['ndcg@10']:.4f} "
            f"diff={diff:+.4f} ci={ci:.4f} tie={out['G0_strength']['tie']}; "
            f"identity: empty==bias {out['G0_identity']['empty_set_equals_frozen_bias']}")
    json.dump(out, open(os.path.join(OUTDIR, "tower_t2.json"), "w"), indent=2)
    log(f"[train] TEST full@10={tm['ndcg@10']:.4f} tail@10={tm['tail_ndcg@10']:.4f} -> tower_t2.json")


# =============================================================================================
# OFFLINE COLD EVAL  (--eval_cold <ckpt>: READ-ONLY -- loads a checkpoint, prints one line, exits;
#                     touches NO training state / checkpoint files; safe to run against a live run)
# =============================================================================================
def eval_cold_mode(args):
    ckpt = args.eval_cold
    if not os.path.exists(ckpt):
        raise SystemExit(f"[eval_cold] checkpoint not found: {ckpt}")
    meta = M.load_meta(PROC); ni = meta["n_items"]
    train_mat = M.load_train(ni, PROC)
    head_mask, cnt = compute_head_mask(train_mat, ni)
    unique_uid, tr_set, vd_set, te_set, n_train, raw, show2id, usid = reproduce_partition()
    L_val, _ = build_graded_eval_matrix(raw, unique_uid, show2id, usid, "validation")
    va_tr, va_te = M.load_val(ni, PROC)
    Lk2 = truncate_graded(L_val, 2, COLD_SEED)
    Lk8 = truncate_graded(L_val, 8, COLD_SEED + 1)
    from scipy import sparse
    L0 = sparse.csr_matrix(L_val.shape, dtype=np.float32)            # EMPTY fold-in (intercept eval)
    enc, decoder, teacher, params, groups = build_model(args, ni, cnt)
    blob = torch.load(ckpt, map_location="cpu")                      # READ-ONLY
    enc.load_state_dict(blob["enc"]); decoder.load_state_dict(blob["decoder"])
    enc.eval()
    Wd_, bd_ = decoder.weight.detach(), decoder.bias.detach()
    full = M.evaluate(make_graded_predict_fn(enc, Wd_, bd_, L_val), va_tr, va_te,
                      batch_size=500, head_mask=head_mask)
    c2 = cold_full10(enc, Wd_, bd_, Lk2, va_tr, va_te, head_mask)
    c8 = cold_full10(enc, Wd_, bd_, Lk8, va_tr, va_te, head_mask)
    c0 = cold_full10(enc, Wd_, bd_, L0, va_tr, va_te, head_mask)
    znorm, rho = report_empty_set(enc, Wd_, bd_, cnt, "eval_cold")
    log(f"[eval_cold] ckpt={os.path.basename(ckpt)} ep={blob.get('epoch', '?')} "
        f"VAL full@10={full['ndcg@10']:.4f} tail@10={full['tail_ndcg@10']:.4f} "
        f"coldk8={c8:.4f} coldk2={c2:.4f} cold0={c0:.4f} |z_empty|={znorm:.4f} "
        f"vs bar {G0_BAR:.4f} ({full['ndcg@10'] - G0_BAR:+.4f})")


# =============================================================================================
# DRY RUN  (real data: leak asserts + one fwd/bwd + calibrated epoch-time estimate; NO training)
# =============================================================================================
def dry_run(args):
    log(f"[DRY] real-data leak asserts + 1 fwd/bwd + epoch-time estimate (teacher={args.teacher})")
    meta = M.load_meta(PROC); ni = meta["n_items"]
    train_mat = M.load_train(ni, PROC)
    _, cnt = compute_head_mask(train_mat, ni)
    unique_uid, tr_set, vd_set, te_set, n_train, raw, show2id, usid = reproduce_partition()
    ncal = max(args.dry_users, 200)
    users = build_train_profiles(raw, tr_set, show2id, max_users=ncal)
    assert all(0 <= s < ni for u in users for s in u["items"]), "sid out of vocab range (LEAK)"
    log("[DRY] token-vocab range assert PASS (all sids in [0, n_items))")

    enc, decoder, teacher, params, groups = build_model(args, ni, cnt)
    report_empty_set(enc, decoder.weight.detach(), decoder.bias.detach(), cnt, "DRY-init")
    opt = torch.optim.AdamW(groups, lr=args.lr)
    if args.arch == "i25":                                   # param-group asserts (arch-gated)
        want = 2 if args.train_decoder else 1
        assert len(opt.param_groups) == want, f"i25 groups: expected {want}, got {len(opt.param_groups)}"
        log(f"[DRY] param-group asserts PASS (i25): {want} group(s), train_decoder={args.train_decoder}")
    elif args.teacher == "warm_init":                        # stabilization patch asserts
        lrs = [g.get("lr", args.lr) for g in opt.param_groups]
        assert len(opt.param_groups) == 2 and abs(lrs[1] - args.lr * args.warm_lr_scale) < 1e-12, \
            f"warm param groups wrong: lrs={lrs}"
        log(f"[DRY] param-group asserts PASS: fast lr={lrs[0]} slow lr={lrs[1]}")

    # one fwd/bwd on the first 200 users
    rng = np.random.default_rng(0)
    exs = [(i, make_input_target(users[i], rng)) for i in range(min(200, len(users)))]
    exs = [(i, e) for i, e in exs if e is not None]
    if args.arch == "i25" or (args.teacher == "warm_init" and not args.full_kd):
        lam0 = 0.0                           # pinned: pure rank loss (i25 / warm mode)
    elif args.full_kd:
        lam0 = args.full_kd_w
    else:
        lam0 = lam_anneal(0, args.lambda_z, args.anneal_epochs)
    loss, nll_v, mse_v, ff_v = batch_loss(enc, decoder, teacher, exs, ni, args, lam0)
    opt.zero_grad(); loss.backward(); opt.step()
    log(f"[DRY] one fwd/bwd on {len(exs)} users OK: NLL={nll_v:.4f} zMSE={mse_v:.4f} lam_glob={lam0:.3f}")
    opt_ids = {id(p) for grp in opt.param_groups for p in grp["params"]}
    if args.arch == "i25":
        assert not enc.item_emb.weight.requires_grad and id(enc.item_emb.weight) not in opt_ids, \
            "i25 identities must be frozen + excluded from the optimizer"
        if args.train_decoder:
            assert decoder.weight.requires_grad and id(decoder.weight) in opt_ids \
                and len(opt.param_groups) == 2, "i25 arm B: decoder must be in the slow group"
        else:
            assert not decoder.weight.requires_grad and id(decoder.weight) not in opt_ids, \
                "i25 arm A: decoder must be frozen + excluded"
        assert teacher is None, "i25 must not carry a loss-path teacher (native enc is in the forward)"
        log(f"[DRY] i25 asserts PASS (identities frozen; decoder arm "
            f"{'B trainable-slow' if args.train_decoder else 'A frozen'}; no loss teacher)")
    elif args.teacher == "recvae":
        assert not decoder.weight.requires_grad and not decoder.bias.requires_grad, "decoder not frozen"
        assert not enc.z0.requires_grad, "z0 not frozen (intercept identity needs z0 fixed at 0)"
        assert args.unfreeze_emb or not enc.item_emb.weight.requires_grad, "item_emb not frozen"
        assert id(decoder.weight) not in opt_ids and id(decoder.bias) not in opt_ids \
            and id(enc.z0) not in opt_ids, "frozen tensor leaked into the optimizer (repair 6)"
        log("[DRY] frozen-geometry asserts PASS (decoder/z0/item_emb frozen + excluded from optimizer)")
    elif args.teacher == "warm_init":
        assert decoder.weight.requires_grad and decoder.bias.requires_grad, "decoder should be TRAINABLE"
        assert enc.item_emb.weight.requires_grad, "item identities should be TRAINABLE"
        assert not enc.z0.requires_grad and id(enc.z0) not in opt_ids, "z0 must stay frozen at 0"
        assert id(decoder.weight) in opt_ids and id(enc.item_emb.weight) in opt_ids, \
            "trainable warm-init tensors missing from the optimizer"
        slow_params = {id(p) for p in opt.param_groups[1]["params"]}
        assert id(enc.item_emb.weight) in slow_params and id(decoder.weight) in slow_params, \
            "warm-started identities/decoder not in the SLOW LR group"
        if args.full_kd:
            assert teacher is not None, "--full_kd must keep the frozen RecVAE in the loss path"
        else:
            assert teacher is None, "warm_init (no full_kd) must not keep the RecVAE in the loss path"
        log("[DRY] warm_init asserts PASS (decoder+identities trainable, SLOW group; z0 frozen; "
            f"teacher-in-loss={args.full_kd})")

    # calibrated epoch-time estimate: time one epoch over `ncal` users, scale to 140768
    lens = np.array([len(u["items"]) for u in users])
    batches_all = make_batches(users, np.argsort(lens))
    enc.train(); t0 = time.time()
    for bat in batches_all:
        exs = [(i, make_input_target(users[i], rng)) for i in bat]
        exs = [(i, e) for i, e in exs if e is not None]
        if not exs:
            continue
        loss, _, _, _ = batch_loss(enc, decoder, teacher, exs, ni, args, lam0)
        opt.zero_grad(); loss.backward(); opt.step()
    dt = time.time() - t0
    est_min = dt * (140768 / len(users)) / 60.0
    log(f"[DRY] timed {len(users)} users ({len(batches_all)} batches) in {dt:.1f}s "
        f"-> EST full epoch @140768 users ~= {est_min:.1f} min ({est_min/60:.1f} h) "
        f"[fwd+bwd only; excludes per-epoch val eval]")
    log("[DRY] COMPLETE")


# =============================================================================================
# SMOKE  (synthetic data + FAKE teacher weights; code-path only, LABELED)
# =============================================================================================
def smoke(args):
    from scipy import sparse
    print("[SMOKE] synthetic tiny data + FAKE RecVAE teacher (code-path only, NOT the canonical split)")
    rng = np.random.RandomState(0)
    ni = 130                         # >100 so NDCG@100 is well-defined
    nu = 240
    users = []
    for _ in range(nu):
        k = rng.randint(6, 80)      # wide: covers interview (<=8), recvae ramp (8-30), full_kd ramp (30-60)
        items = rng.choice(ni, size=k, replace=False).astype(np.int64)
        lvls = rng.randint(0, NLEV, size=k).astype(np.int64)          # ALL bands (graded)
        liked = items[lvls >= 7]
        if len(liked) < 2:
            liked = items[:2]; lvls[:2] = 8
        users.append({"items": items, "levels": lvls, "vals": level_to_sv(lvls), "liked": liked,
                      "disliked": items[lvls <= 4]})       # repair 4: dislike band (<=2.5 stars)

    teacher_override = None
    if args.arch == "i25" or args.teacher in ("recvae", "warm_init"):
        # fake teacher/init source: random-weight RecVAE saved+loaded through the REAL load path
        args.t_hidden, args.t_latent = 24, 16
        fake = R.RecVAE(args.t_hidden, args.t_latent, ni)
        tmp = os.path.join(CKPT_DIR, "_smoke_recvae.pt"); os.makedirs(CKPT_DIR, exist_ok=True)
        torch.save({"model": fake.state_dict(), "state": {"epoch": 1, "best": 0.0, "best_state": None}}, tmp)
        teacher_override = load_recvae_teacher(ni, path=tmp, hidden=args.t_hidden, latent=args.t_latent)
        os.remove(tmp)
    cnt = np.ones(ni)
    if getattr(args, "concept_tokens", False):
        args.n_concepts = 6                                   # toy concept vocab for the C-full smoke
    enc, decoder, teacher, params, groups = build_model(args, ni, cnt, teacher_override=teacher_override)
    znorm, rho = report_empty_set(enc, decoder.weight.detach(), decoder.bias.detach(), None, "SMOKE-init")
    assert znorm < 1e-6, "gated intercept identity broken (enc(empty) != 0 at init; gate g(0) leak?)"
    if args.arch == "i25":
        # i25 zero-init identity: z == native_z BIT-EXACT at init (rho final layer zero-init)
        r0 = np.random.default_rng(7)
        exs0 = [(j, make_input_target(users[j], r0)) for j in range(8)]
        exs0 = [(j, e) for j, e in exs0 if e is not None]
        ii0, vv0, pp0, ll0 = pack_tokens([(e[0], e[1], e[2]) for _, e in exs0])
        enc.eval()
        with torch.no_grad():
            n0 = (~pp0).sum(-1, keepdim=True).float()
            expect = enc.anchor(n0) * enc.native_z(ii0, pp0, ll0)     # S1: init identity = a(n)*native
            assert torch.equal(enc(ii0, vv0, pp0, ll0), expect), \
                "i25 zero-init identity broken (z != a(n)*native_z at init)"
        print("[SMOKE] i25 zero-init identity PASS: z == a(n)*native_z BIT-EXACT at init (S1 anchored)")
        # S1 anchor-gate properties: a(0)=0 exactly, strictly monotone, ~1 by n~16 at init
        with torch.no_grad():
            nn_ = torch.arange(0, 33, dtype=torch.float32).unsqueeze(1)
            av = enc.anchor(nn_).squeeze(1)
        assert float(av[0]) == 0.0, "a(0) != 0 (empty-set identity would break)"
        assert bool((av[1:] > av[:-1]).all()), "a(n) not strictly monotone"
        assert float(av[16]) >= 0.97, f"a(16)={float(av[16]):.3f} (expected ~0.98 ramp at init)"
        print(f"[SMOKE] anchor gate PASS: a(0)=0, monotone, a(2)={float(av[2]):.3f} "
              f"a(8)={float(av[8]):.3f} a(16)={float(av[16]):.3f}")
        # S5 dedup assert must fire on a duplicated sid
        try:
            pack_tokens([(np.array([3, 3, 5]), np.array([8, 8, 9]), level_to_sv(np.array([8, 8, 9])))])
            raise RuntimeError("dedup assert did NOT fire on duplicate sids")
        except AssertionError:
            print("[SMOKE] dedup assert PASS: duplicate sids in a token set hard-fail (S5)")
    if args.arch == "i25" or args.teacher in ("recvae", "warm_init"):
        with torch.no_grad():
            sc0 = (enc(torch.zeros((1, 1), dtype=torch.long), torch.zeros((1, 1)),
                       torch.ones((1, 1), dtype=torch.bool), torch.zeros((1, 1), dtype=torch.long))
                   @ decoder.weight.T + decoder.bias)[0]
        assert torch.equal(sc0, decoder.bias), "enc(empty) does not decode BIT-EXACTLY to the bias"
        print("[SMOKE] intercept identity PASS: enc(empty) decodes BIT-EXACTLY to the decoder bias "
              "(gate g(0)=0 + frozen z0=0 -- holds at every training step; in warm_init the bias itself "
              "is trainable, the identity tracks the CURRENT bias)")
        if args.sign_prior:
            g = enc.gamma.weight.detach()
            for lvl, want in enumerate(SIGN_GAMMA_V2):
                assert torch.allclose(g[lvl], torch.full_like(g[lvl], want)), \
                    f"gamma(level {lvl}) != {want}"
            assert SIGN_GAMMA_V2[0] == -1.0 and SIGN_GAMMA_V2[NLEV - 1] == 1.0   # ends pinned
            assert all(SIGN_GAMMA_V2[l] == 1.0 for l in range(LIKE_MIN_LEVEL, NLEV)), \
                "all like levels must init at full +1 (pb2 parity)"
            assert float(enc.beta.weight.detach().abs().max()) < 1e-12, "beta not zero-init"
            print(f"[SMOKE] sign init v2 PASS: gamma={SIGN_GAMMA_V2} "
                  f"(likes full +1, neutral +0.25, dislikes graded negative), beta=0")

    t_idx = t_prob = None
    if args.alpha_kd > 0:
        Bt = (rng.randn(ni, ni) * 0.1).astype(np.float32); np.fill_diagonal(Bt, 0.0)
        tmp = os.path.join(CKPT_DIR, "_smoke_teacherB.npy"); os.makedirs(CKPT_DIR, exist_ok=True)
        np.save(tmp, Bt)
        t_idx, t_prob = precompute_teacher(users, tmp, ni, topk=10, temp=2.0)
        os.remove(tmp)
        print("[SMOKE] EDLAE KD teacher precompute path exercised (synthetic B)")

    opt = torch.optim.AdamW(groups, lr=1e-3)
    if args.arch == "i25":
        want_groups = 2 if args.train_decoder else 1
        assert len(opt.param_groups) == want_groups, \
            f"i25 arm {'B' if args.train_decoder else 'A'} should give {want_groups} param group(s)"
        print(f"[SMOKE] param-group PASS (i25): {len(opt.param_groups)} group(s), "
              f"train_decoder={args.train_decoder}")
    elif args.teacher == "warm_init":
        assert len(opt.param_groups) == 2, "warm_init should give 2 param groups"
        print(f"[SMOKE] param-group PASS: fast lr={opt.param_groups[0]['lr']} "
              f"slow lr={opt.param_groups[1]['lr']} (warm_lr_scale={args.warm_lr_scale})")
    # patch 4 unit-smoke: synthetic NLL sequence must trigger the rescue on the 2nd consecutive rise
    gtest = {"min": float("inf"), "streak": 0}
    lr_before = [g["lr"] for g in opt.param_groups]
    fired = [nll_guard_step(gtest, v, opt, enc, decoder, None) for v in (5.0, 4.0, 4.6, 4.7)]
    assert fired == [False, False, False, True], f"nll_guard sequence wrong: {fired}"
    assert all(abs(g["lr"] - 0.5 * l0) < 1e-12 for g, l0 in zip(opt.param_groups, lr_before)), \
        "nll_guard did not halve LRs"
    for g, l0 in zip(opt.param_groups, lr_before):
        g["lr"] = l0                                        # restore for the real smoke epochs
    print("[SMOKE] nll_guard PASS: fired on 2nd consecutive >10% rise, halved all LRs (restored)")
    # 2026-07-23 fix unit-smoke: a rescue must RESET the patience counter (loop continues), not exit
    bad, stop = early_stop_step(3, improved=False, rescued=True, patience=4)
    assert (bad, stop) == (0, False), f"rescue at bad=3/patience=4 must continue, got {(bad, stop)}"
    bad, stop = early_stop_step(3, improved=False, rescued=False, patience=4)
    assert (bad, stop) == (4, True), "no-rescue path must still early-stop at patience"
    assert early_stop_step(3, improved=True, rescued=False, patience=4) == (0, False)
    print("[SMOKE] early_stop_step PASS: guard rescue resets patience (loop CONTINUES after rescue)")
    lens = np.array([len(u["items"]) for u in users])
    batches_all = make_batches(users, np.argsort(lens))
    # p_interview schedule parse smoke
    assert parse_p_interview("0.5") == (0.5, 0.5) and parse_p_interview("0.1:0.5") == (0.1, 0.5)
    assert abs(p_interview_at(5, 0.1, 0.5, 10) - 0.3) < 1e-12
    print("[SMOKE] p_interview schedule PASS ('0.5' const; '0.1:0.5' linear over anneal_epochs)")
    p0, p1 = parse_p_interview(args.p_interview)
    E0 = enc.item_emb.weight.detach().clone()
    W0 = decoder.weight.detach().clone()
    for ep in range(3):
        enc.train(); r = np.random.default_rng(ep)
        if args.teacher == "warm_init" and not args.full_kd:
            lam_glob = 0.0                   # pinned (mirrors train())
        elif args.full_kd:
            lam_glob = args.full_kd_w
        else:
            lam_glob = lam_anneal(ep, args.lambda_z, args.anneal_epochs)
        p_int = p_interview_at(ep, p0, p1, args.anneal_epochs)
        last = (float("nan"), float("nan"))
        for bat in batches_all:
            exs = [(i, make_input_target(users[i], r, p_int=p_int)) for i in bat]
            exs = [(i, e) for i, e in exs if e is not None]
            if not exs:
                continue
            loss, nll_v, mse_v, ff_v = batch_loss(enc, decoder, teacher, exs, ni, args, lam_glob, t_idx, t_prob)
            opt.zero_grad(); loss.backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(params, args.clip)
            opt.step()
            last = (nll_v, mse_v)
        if args.teacher == "warm_init" and not args.full_kd:
            assert lam_glob == 0.0, "warm mode without full_kd must pin lam_glob=0"
        print(f"[SMOKE] ep{ep+1} last-batch NLL={last[0]:.4f} zMSE={last[1]:.4f} lam_glob={lam_glob:.3f}")
    if args.arch == "i25":
        assert torch.equal(enc.item_emb.weight, E0), "i25 frozen identities CHANGED during training"
        if args.train_decoder:
            assert not torch.equal(decoder.weight, W0), "i25 arm B decoder did NOT move"
            print("[SMOKE] i25 arm B PASS: decoder moved (slow group); identities frozen")
        else:
            assert torch.equal(decoder.weight, W0), "i25 arm A frozen decoder CHANGED"
            print("[SMOKE] i25 arm A PASS: decoder + identities bit-identical after 3 epochs")
        # sum-pool permutation invariance (post-training, rho nonzero)
        u0 = users[0]; k0n = len(u0["items"])
        perm = np.random.default_rng(11).permutation(k0n)
        b1 = pack_tokens([(u0["items"], u0["levels"], u0["vals"])])
        b2 = pack_tokens([(u0["items"][perm], u0["levels"][perm], u0["vals"][perm])])
        enc.eval()
        with torch.no_grad():
            z1 = enc(*b1); z2 = enc(*b2)
        assert torch.allclose(z1, z2, atol=1e-5), "sum-pool permutation invariance broken"
        print("[SMOKE] i25 permutation invariance PASS (token order changes z by < 1e-5)")
        # dislike-in-phi: flipping one liked token to hated must CHANGE z (post-training)
        lv3 = u0["levels"].copy(); lv3[0] = 0                # 0.5 stars
        b3 = pack_tokens([(u0["items"], lv3, level_to_sv(lv3))])
        with torch.no_grad():
            z3 = enc(*b3)
        assert not torch.allclose(z1, z3, atol=1e-6), "dislike level flip did not change z (phi dead?)"
        print("[SMOKE] i25 dislike-in-phi PASS: like->hate flip moves z")
        # ---- ARM C-FULL smoke (--concept_tokens): channel asserts ----
        if args.concept_tokens:
            from concept_fold import build_member_matrix, sel_concepts_from_items
            nc = int(args.n_concepts)
            # (1) NO-OP BIT-IDENTITY: item-only forward identical to an nc=0 build with shared weights
            enc0 = I25Encoder(ni, teacher_override, d_lat=enc.d_out, token_mode=args.token,
                              n_concepts=0)
            enc0.load_state_dict(enc.state_dict(), strict=False)          # concept_emb key ignored
            enc0.eval()
            with torch.no_grad():
                assert torch.equal(enc(*b1), enc0(*b1)), \
                    "concept channel broke item-only bit-identity (G0 protection violated)"
            print("[SMOKE] C-full no-op PASS: item-only forward BIT-IDENTICAL to the nc=0 build")
            # (2) native anchor NEVER sees concepts
            u0c = users[0]
            ids_c = np.concatenate([u0c["items"][:4], [ni + 1, ni + 3]]).astype(np.int64)
            lvs_c = np.concatenate([u0c["levels"][:4], [9, 8]]).astype(np.int64)
            bA = pack_tokens([(u0c["items"][:4], u0c["levels"][:4], u0c["vals"][:4])])
            bC = pack_tokens([(ids_c, lvs_c, level_to_sv(lvs_c))])
            with torch.no_grad():
                nzA = enc.native_z(bA[0], bA[2], bA[3]); nzC = enc.native_z(bC[0], bC[2], bC[3])
                assert torch.equal(nzA, nzC), "concept tokens leaked into the frozen native anchor"
                zC = enc(*bC); zA = enc(*bA)
            assert not torch.equal(zC, zA), "concept tokens inert in the forward (channel dead)"
            print("[SMOKE] C-full anchor-isolation PASS: native_z ignores concepts; forward uses them")
            # (3) sel_value_to_level convention: [0.25,1] -> {6..9} positive band
            assert sel_value_to_level(0.25) == 6 and sel_value_to_level(1.0) == 9
            assert all(6 <= sel_value_to_level(v) <= 9 for v in (0.3, 0.5, 0.7, 0.9))
            print("[SMOKE] C-full value map PASS: SEL [0.25,1] -> levels {6..9}")
            # (4) item-masked curriculum: revealed concepts' member items dropped; targets leak-free
            rngS = np.random.RandomState(3)
            membS = {t: np.sort(rngS.choice(ni, size=rngS.randint(20, 40), replace=False))
                     for t in range(nc)}
            MmS = build_member_matrix(membS, list(range(nc)), ni)
            cntS = np.ones(ni)
            grateS = (cntS @ np.asarray(MmS.todense())) / cntS.sum()
            msetS = {i: membS[i] for i in range(nc)}
            hits = 0
            for tr_i in range(200):
                uS = users[tr_i % len(users)]
                e = make_concept_example(uS, np.random.default_rng(tr_i), MmS, grateS, msetS, ni)
                if e is None:
                    continue
                inp, lvv, svv, tgt, negs = e
                cmask = inp >= ni
                assert cmask.any(), "concept example contains no concept token"
                assert (lvv[cmask] >= 6).all(), "concept token level outside the positive band"
                for cid in inp[cmask] - ni:
                    assert not np.isin(inp[~cmask], msetS[int(cid)]).any(), \
                        "member items of a revealed concept leaked into the item input"
                assert len(np.intersect1d(inp[~cmask], tgt)) == 0, "target leaked into input"
                hits += 1
            assert hits > 20, f"too few usable concept examples ({hits}/200)"
            print(f"[SMOKE] C-full curriculum PASS ({hits}/200 usable; member-drop + leak asserts)")
            # (5) param delta
            extra = enc.concept_emb.weight.numel()
            print(f"[SMOKE] C-full param delta (toy dims): +{extra:,}; REAL dims: 1031 x 200 = "
                  f"+206,200 on 838,250 -> 1,044,450 trainable")
    elif args.teacher == "recvae":
        assert torch.equal(decoder.weight, teacher_override.decoder.weight) and \
               torch.equal(decoder.bias, teacher_override.decoder.bias), \
               "frozen decoder CHANGED during training"
        assert torch.equal(enc.item_emb.weight, E0) or args.unfreeze_emb, \
               "frozen item identities CHANGED during training"
        print("[SMOKE] frozen-geometry immutability PASS (decoder + item identities bit-identical "
              "after 3 epochs)")
    elif args.teacher == "warm_init":
        assert not torch.equal(decoder.weight, W0), "warm_init decoder did NOT move (should be trainable)"
        assert not torch.equal(enc.item_emb.weight, E0), "warm_init identities did NOT move"
        print("[SMOKE] warm_init trainability PASS (decoder + item identities moved after 3 epochs)")
    if args.arch == "i25" or args.teacher in ("recvae", "warm_init"):
        # repair 5: the intercept must STILL be bit-exact AFTER training (gate property, not init;
        # in warm_init the identity tracks the CURRENT trainable bias)
        enc.eval()
        with torch.no_grad():
            sc1 = (enc(torch.zeros((1, 1), dtype=torch.long), torch.zeros((1, 1)),
                       torch.ones((1, 1), dtype=torch.bool), torch.zeros((1, 1), dtype=torch.long))
                   @ decoder.weight.T + decoder.bias)[0]
        assert torch.equal(sc1, decoder.bias), "intercept identity LOST after training (gate broken)"
        print("[SMOKE] post-training intercept PASS: enc(empty) STILL decodes bit-exactly to the "
              "(current) bias")

    # ---- w_neg boundedness smoke (2026-07-23 divergence fix): LEARNABLE 2-cluster data WITH dislikes,
    # trained several epochs with the clamped dislike term. Asserts: every loss finite, like-NLL DROPS
    # (the unclamped form made like-NLL RISE once the softmax sharpened), floor fraction reported.
    print("[SMOKE] w_neg boundedness: 2-cluster synthetic with dislikes, 8 epochs, clamped neg term")
    cl_users = []
    for i in range(60):
        lik = np.arange(0, 20) if i % 2 == 0 else np.arange(100, 120)
        dis = np.arange(100, 120) if i % 2 == 0 else np.arange(0, 20)
        levels = np.concatenate([np.full(20, 9), np.full(20, 0)]).astype(np.int64)
        cl_users.append({"items": np.concatenate([lik, dis]).astype(np.int64), "levels": levels,
                         "vals": level_to_sv(levels), "liked": lik.astype(np.int64),
                         "disliked": dis.astype(np.int64)})
    enc2 = SetEncoder(ni, d=64, d_out=16, d_emb=16, m=4, nlayers=1)
    apply_sign_prior(enc2)
    dec2 = nn.Linear(16, ni)
    opt2 = torch.optim.AdamW([p for p in list(enc2.parameters()) + list(dec2.parameters())
                              if p.requires_grad], lr=1e-3)
    ep_nlls = []; last_ff = float("nan")
    for ep in range(8):
        r2 = np.random.default_rng(100 + ep); tot_n = 0.0; nb2 = 0
        for st in range(0, len(cl_users), 30):
            exs2 = [(j, make_input_target(cl_users[j], r2)) for j in range(st, min(st + 30, len(cl_users)))]
            exs2 = [(j, e) for j, e in exs2 if e is not None]
            if not exs2:
                continue
            l2, n2, _, f2 = batch_loss(enc2, dec2, None, exs2, ni, args, 0.0)
            assert torch.isfinite(l2), f"loss not finite at ep{ep} (boundedness broken)"
            opt2.zero_grad(); l2.backward(); opt2.step()
            tot_n += n2; nb2 += 1; last_ff = f2
        ep_nlls.append(tot_n / max(nb2, 1))
    print(f"[SMOKE] w_neg run: like-NLL {ep_nlls[0]:.4f} -> {ep_nlls[-1]:.4f}; "
          f"final neg-at-floor frac={last_ff:.2f}")
    assert ep_nlls[-1] < ep_nlls[0], \
        f"like-NLL rose under w_neg ({ep_nlls[0]:.4f}->{ep_nlls[-1]:.4f}): divergence NOT fixed"
    print("[SMOKE] w_neg boundedness PASS: all losses finite, like-NLL decreased with dislikes active")
    # direct saturation check: crush dislike logits far below the floor -> floor_frac must hit 1.0
    with torch.no_grad():
        dec2.bias[np.arange(100, 120)] -= 50.0               # dislikes now ~e^-50 below uniform
    r2 = np.random.default_rng(999)
    exs2 = [(j, make_input_target(cl_users[j], r2)) for j in range(0, 60, 2)]   # cluster A (dislikes 100+)
    exs2 = [(j, e) for j, e in exs2 if e is not None]
    _, _, _, ff_sat = batch_loss(enc2, dec2, None, exs2, ni, args, 0.0)
    assert ff_sat == 1.0, f"clamp did not saturate on crushed dislikes (floor_frac={ff_sat})"
    print("[SMOKE] w_neg saturation PASS: crushed dislikes report floor_frac=1.0 (zero further gradient)")

    # eval path smoke: synthetic graded fold-in matrix + head_mask through metrics.evaluate
    tr = sparse.csr_matrix((np.ones(600), (rng.randint(0, 60, 600), rng.randint(0, ni, 600))),
                           shape=(60, ni), dtype=np.float32)
    tr.data[:] = rng.randint(7, NLEV, size=tr.nnz).astype(np.float32) + 1.0    # levels 7-9 (likes), +1
    bin_tr = (tr > 0).astype(np.float32)
    te = sparse.csr_matrix((np.ones(200), (rng.randint(0, 60, 200), rng.randint(0, ni, 200))),
                           shape=(60, ni), dtype=np.float32)
    te = (te > 0).astype(np.float32); te.data[:] = 1.0
    hm, _ = compute_head_mask(bin_tr, ni)
    predict = make_graded_predict_fn(enc, decoder.weight.detach(), decoder.bias.detach(), tr)
    res = M.evaluate(predict, bin_tr, te, batch_size=20, head_mask=hm)
    print(f"[SMOKE] eval path OK  full@10={res['ndcg@10']:.4f} tail@10={res['tail_ndcg@10']:.4f} "
          f"ndcg@100={res['ndcg@100']:.4f}")

    # ---- cold-val smoke (2026-07-23): fixed-subset truncation + cold eval path + eval_cold contract ----
    t2a = truncate_graded(tr, 2, COLD_SEED); t2b = truncate_graded(tr, 2, COLD_SEED)
    assert (t2a != t2b).nnz == 0, "truncation not deterministic under the fixed seed"
    for r in range(tr.shape[0]):
        want = min(2, tr.indptr[r + 1] - tr.indptr[r])
        assert t2a.indptr[r + 1] - t2a.indptr[r] == want, f"row {r} truncated wrong"
    t8 = truncate_graded(tr, 8, COLD_SEED + 1)
    c2s = cold_full10(enc, decoder.weight.detach(), decoder.bias.detach(), t2a, bin_tr, te, hm)
    c8s = cold_full10(enc, decoder.weight.detach(), decoder.bias.detach(), t8, bin_tr, te, hm)
    L0s = sparse.csr_matrix(tr.shape, dtype=np.float32)
    c0s = cold_full10(enc, decoder.weight.detach(), decoder.bias.detach(), L0s, bin_tr, te, hm)
    assert np.isfinite(c2s) and np.isfinite(c8s) and np.isfinite(c0s)
    print(f"[SMOKE] cold-val path PASS: deterministic k-subsets; coldk2={c2s:.4f} coldk8={c8s:.4f} "
          f"cold0(empty)={c0s:.4f}")
    # eval_cold ckpt contract: save read-only-style blob, reload into FRESH modules, scores identical
    tmpck = os.path.join(CKPT_DIR, "_smoke_evalcold.pt")
    torch.save({"enc": enc.state_dict(), "decoder": decoder.state_dict(), "epoch": 3}, tmpck)
    if args.arch == "i25":
        enc3 = I25Encoder(ni, teacher_override, d_lat=enc.d_out, token_mode=args.token,
                          n_concepts=getattr(enc, "nc", 0))
    else:
        enc3 = SetEncoder(ni, d=D, d_out=enc.d_out, d_emb=enc.item_emb.weight.shape[1],
                          token_mode=args.token, norm_feat=enc.norm_feat)
    dec3 = nn.Linear(enc.d_out, ni)
    blob3 = torch.load(tmpck, map_location="cpu")
    enc3.load_state_dict(blob3["enc"]); dec3.load_state_dict(blob3["decoder"]); enc3.eval()
    os.remove(tmpck)
    c2r = cold_full10(enc3, dec3.weight.detach(), dec3.bias.detach(), t2a, bin_tr, te, hm)
    assert abs(c2r - c2s) < 1e-9, f"eval_cold reload mismatch: {c2r} vs {c2s}"
    print("[SMOKE] eval_cold contract PASS: fresh-module reload reproduces scores bit-for-bit")
    print("[SMOKE] COMPLETE (paths: graded tokens, interview-regime subsets, frozen-geometry latent "
          "distill, EDLAE KD hook, canonical eval)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag", default="tower_t2")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--token", choices=["film", "mlp"], default="film")
    # architecture revision 2026-07-23 (i25 residual fold; FOLD_MASTER F7)
    ap.add_argument("--arch", choices=["i25", "attn"], default="i25",
                    help="i25 = residual fold on the frozen native RecVAE encoder (DEFAULT); "
                         "attn = the previous pb2-class attention path (--teacher modes)")
    ap.add_argument("--train_decoder", action="store_true",
                    help="i25 arm B (fallback): decoder trainable at the slow LR group; "
                         "default arm A = frozen RecVAE decoder (G0-identity)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--resume_from_best", action="store_true",
                    help="restart from <tag>_best.pt weights (fresh optimizer at the grouped LRs); "
                         "takes precedence over --resume")
    # stabilization patch (warm_init ep5 collapse, 2026-07-22)
    ap.add_argument("--warm_lr_scale", type=float, default=0.1,
                    help="LR multiplier for the warm-started decoder+identities group (warm_init mode)")
    ap.add_argument("--clip", type=float, default=1.0, help="global grad-norm clip (0 disables)")
    ap.add_argument("--nll_guard", dest="nll_guard", action="store_true", default=True,
                    help="plateau-rescue: halve LRs + reload best if train NLL rises >10% over its "
                         "running min for 2 consecutive epochs (default ON)")
    ap.add_argument("--no_nll_guard", dest="nll_guard", action="store_false")
    # plan-B config (pre-staged 2026-07-22; composable with --teacher warm_init)
    ap.add_argument("--full_kd", action="store_true",
                    help="warm_init plan-B: frozen RecVAE latent term at LARGE subsets only "
                         f"(ramp {FULL_KD_LO}->{FULL_KD_HI}), constant weight full_kd_w, no anneal; "
                         "decoder stays trainable")
    ap.add_argument("--full_kd_w", type=float, default=0.3, help="constant latent weight for --full_kd")
    ap.add_argument("--p_interview", default="0.5",
                    help="interview-regime example fraction; '0.5' constant or 'a:b' linear a->b "
                         "over anneal_epochs (curriculum)")
    # cold-val additions (2026-07-23)
    ap.add_argument("--no_cold_val", action="store_true",
                    help="skip the per-epoch cold (k=2/k=8 fixed-subset) val evals")
    # ARM C-FULL (2026-07-24): concept tokens trained INTO the tower. Default OFF -- t2final provenance
    # untouched; i25-only.
    ap.add_argument("--concept_tokens", action="store_true",
                    help="ARM C-FULL: add a trainable concept-token channel (ids ni+c, FiLM-fused, "
                         "same phi pool; frozen native anchor NEVER sees concepts)")
    ap.add_argument("--p_concept_ex", type=float, default=0.5,
                    help="fraction of training examples drawn from the concept curriculum "
                         "(rest = plain item examples; keeps the item pathway fed)")
    ap.add_argument("--signed_concepts", action="store_true",
                    help="SIGNED four-band SEL+VAL concept channel (DESIGN_SIGNED_CONCEPTS): "
                         "m<=16, dislike levels 0..4 in-envelope, C_NEG cap; requires "
                         "--concept_tokens")
    ap.add_argument("--select_cold", action="store_true",
                    help="AUTHOR SELECTION RULE (2026-07-24): best checkpoint = max cold composite "
                         "(mean coldk2,coldk8) SUBJECT TO full@10 >= native-init - G0_TIE_CI; patience "
                         "runs on this criterion. Requires cold_val. Default off (full@10-primary).")
    ap.add_argument("--eval_cold", default=None, metavar="CKPT",
                    help="OFFLINE READ-ONLY: load CKPT, print full + coldk2/k8 + empty-set val line, exit")
    ap.add_argument("--max_users", type=int, default=0, help="cap MATERIALISED users (dev only; 0=all)")
    ap.add_argument("--dry_users", type=int, default=2000, help="users to time for the epoch estimate")
    # teacher-mode revision (2026-07-22)
    ap.add_argument("--teacher", choices=["warm_init", "recvae", "none"], default="warm_init",
                    help="warm_init = TRAINABLE decoder/identities from RecVAE init, no latent KD "
                         "(DEFAULT; probe-FAIL escalation); recvae = frozen T1 geometry + latent "
                         "distill; none = from-scratch arm")
    ap.add_argument("--lambda_z", type=float, default=0.5,
                    help="INITIAL global latent-KD weight; annealed ->0 over anneal_epochs (repair 1)")
    ap.add_argument("--anneal_epochs", type=int, default=10,
                    help="epochs over which lambda_z anneals to 0 (KD = warm-start only)")
    ap.add_argument("--w_neg", type=float, default=0.1,
                    help="weight of held-dislike explicit negatives in the rank loss (repair 4)")
    ap.add_argument("--unfreeze_emb", action="store_true", help="fallback arm: train the item embeddings")
    ap.add_argument("--no_sign_prior", dest="sign_prior", action="store_false", default=True,
                    help="ablation: skip the signed FiLM-beta init (addendum 2026-07-22)")
    ap.add_argument("--t_hidden", type=int, default=600)
    ap.add_argument("--t_latent", type=int, default=200)
    ap.add_argument("--ablate_binarized", action="store_true",
                    help="G3 canary: collapse INPUT levels to a constant like-level (presence-only)")
    # T3' EDLAE score-CE hook (default OFF)
    ap.add_argument("--alpha_kd", type=float, default=0.0, help="EDLAE-teacher KD weight (0=off)")
    ap.add_argument("--teacher_npy", default=None, help="EDLAE B .npy (edlae.py --export_B)")
    ap.add_argument("--kd_topk", type=int, default=1000)
    ap.add_argument("--kd_temp", type=float, default=2.0)
    args = ap.parse_args()
    if args.concept_tokens:
        assert args.arch == "i25", "--concept_tokens is i25-only (Arm C-full)"
    if args.signed_concepts:
        assert args.concept_tokens, "--signed_concepts requires --concept_tokens"
    if args.eval_cold:
        eval_cold_mode(args)
    elif args.smoke:
        smoke(args)
    elif args.dry_run:
        dry_run(args)
    elif args.train:
        train(args)
    else:
        ap.error("one of --smoke / --dry_run / --train / --eval_cold required")


if __name__ == "__main__":
    main()
