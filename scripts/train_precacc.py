r"""train_precacc.py -- ANALYTIC per-user PRECISION ACCUMULATOR (Laplace/GGN posterior) on the FROZEN
pbC set-encoder z-space. Adapted from the REVIEWED scripts/train_varhead.py; the learned low-rank head
(Stage-1 VarHead) FAILED G1 (Sigma from a pooled summary was content-independent: tr flat 265->317,
directional ratio ~0). Fix: DERIVE Sigma from the evidence that was folded, not estimate it.

MODEL (the only new part -- the MEAN mu is UNTOUCHED, mu = frozen pbC encode_pool of S):
    Lambda(S) = Lambda0 + sum_{j in S} alpha_{channel(j), level-bucket(j)} * d_j d_j^T,   Sigma = Lambda^{-1}
    Lambda0   = diag(1 / v0),  v0 = s0 * (empirical full-profile z-variance + vfloor)
                (empirical variance = the salvaged bpool2 anisotropic prior, cached at
                 .cache/set_mn/vhead_sig0_pbC_best.npy -- that file is the VARIANCE)
    d_j (ITEM answer)    = normalize(Wd[item])            (decoder row = exact affinity/GGN curvature dir)
    d_j (CONCEPT answer) = normalize(whiten(centroid_c))  (whiten = center + strip top-1 PC of Wd; the
                            same whitened member-centroid d_c the belief-pool/tree used)
    alpha = a (2 channels x 5 level-buckets) = 10 positive scalars (exp-param); + s0, vfloor, tau2
            => 13 fitted scalars TOTAL. Refusal bucket init ~0.

SIGMA ALGEBRA: Woodbury when m=|S| <= WOOD_MAX (a capacity-free IDENTITY, m x m solves only):
    Sigma = V0 - V0 U (I_m + U^T V0 U)^{-1} U^T V0,   U = [sqrt(alpha_j) d_j],  V0 = diag(v0)
    w'Sigma w, tr(Sigma) from the same m x m Cholesky.  Gated fold uses
    Sigma (Sigma + S I)^{-1} = (I + S Lambda)^{-1} = (diag(1 + S/v0) + S U U^T)^{-1}  (Woodbury again).
When m > WOOD_MAX (full-profile evidence, where Woodbury is the EXPENSIVE side) we Cholesky the dense
512x512 Lambda instead -- still EXACT, just the cheaper exact route for m > d.

ALPHA-FIT = the INCREMENT OBJECTIVE (calibrate REDUCTIONS, not levels). The previous ABSOLUTE
residual-NLL ZEROED concept precision (alpha_conc 0.27->0.013): a good frozen mean leaves no absolute
residual for concept precision to explain (the decoupling trap). Fix, per train user (deterministic
make_pb_example split, seed INC_SEED+uidx): evidence set S = the make_pb_example SUB-interview (heavy
item dropout, 1-32 concepts) -> TARGET s*_i = s_i(mu_full) at the held-liked probe items i = tg, where
mu_full = frozen pbC fold of the user's FULL KNOWN HALF (ALL of u['items']/u['sv'] at their REAL levels
+ the example's revealed concept answers). tg is NEVER folded into any EVIDENCE fold the increments are
computed on (mu(S), mu(S\a)); mu_full is the fixed known-half destination the belief climbs toward
(known-half only -- no held-liked OTHER-half/eval leakage). Because S is a strict sub-interview of the
known half, s* != s_S in general and both Delta_e terms carry signal. For each answer atom a in S:
    ACTUAL   Delta_e_{u,i,a} = (s*_i - s_i(mu(S\a)))^2 - (s*_i - s_i(mu(S)))^2      [frozen, NO alpha,
             precomputed ONCE over ALL train users and cached; the batched leave-one-out folds mu(S\a)
             are the only encoder cost -- the fit loop itself needs NO encoder forward]
    PREDICT  Delta_v_{u,i,a}(alpha) = alpha_a (w_i' Sigma(S\a) d_a)^2 / (1 + alpha_a d_a' Sigma(S\a) d_a)
                                    = alpha_a (w_i' Sigma(S)  d_a)^2 / (1 - alpha_a d_a' Sigma(S)  d_a)
             (EXACT Sherman-Morrison downdate identity: ONE full-S SigmaOps per user yields every
              leave-one-out prediction; the denominator is in (0,1] because Lambda(S\a) is SPD)
    L = mean_u mean_{i,a} Huber_beta( Delta_v - Delta_e );  alpha (2x5) + s0 + vfloor fitted; tau2 INERT
    (kept in the state dict for compat -- the increment loss has no observation-noise term).
Sign is right because both deltas describe the SAME added answer: informative answer -> Delta_e>0 ->
the fit wants Delta_v>0 -> alpha>0. The good frozen mean is now the SOURCE of credit (large Delta_e),
not the destroyer. Delta_e is SIGNED (term2 > 0 in general, since mu_full != mu(S)): an atom that moves
mu(S) TOWARD the full-known-half fold earns positive credit; an uninformative atom earns ~0; a
misleading atom earns negative credit -- the objective can now assign zero/negative credit, which the
old subsample-target form (s* = s_S, term2 == 0, Delta_e >= 0 identically) could not.

PRE-FIT SIGN PROOF (mandatory gate, seconds, BEFORE the fit): at alpha_conc=0, dL/dalpha_conc < 0 iff
    G = sum_{concept triples} clamp(Delta_e_{u,i,c}, -beta, beta) * (w_i' Sigma(S\c) d_c)^2  >  0
(the clamp IS the Huber gradient at residual -Delta_e: smooth-L1' is x/beta inside |x|<beta, sign(x)
outside, so the exact statistic weights each triple by clamp(De,-beta,beta) -- with Delta_e now signed
this matters; the unclamped sum is NOT the gradient sign. With the concept channel zeroed, concept
atoms add NO precision, so Sigma(S) == Sigma(S\c) exactly).
G is logged and ASSERTED > 0 -- guarantees alpha_conc leaves zero and cannot collapse like the absolute
objective. G <= 0 => bug/data problem: STOP and report, do not train.

GATES (reused from train_varhead, adjusted for the analytic Sigma):
  G0  eval_student on mu bit-identical to 0.4946/0.3372 (trivial -- mu untouched -- but asserted), zero
      trunk+decoder drift.
  G1a tr(Sigma) strictly decreasing over NESTED evidence prefixes k in {0,1,2,4,8,16,32,full}. MUST hold
      BY CONSTRUCTION (adding PSD precision cannot inflate variance) -> any violation is a CODE BUG: assert.
  G1b folding concept c ("loved") cuts RELATIVE variance along d_c >= 2x vs other concepts' directions.
      Also by construction (precision added exactly along d_c) -> violation = CODE BUG: assert.
  G1c calibration Spearman rho(sqrt(w'Sigma w), held-item NLL) -- DECISIVE, bar >= 0.25 (the failed
      learned head scored 0.219; the analytic Sigma must beat it).
  G2  (decisive) cold-start per-step q=0..8, truthful answers, REAL decoder, FULL+TAIL, THREE selection
      orderings from the SAME answerability-masked pool:
        (i)  SPINE = DIRECT informativeness: value(c) = mean over the 4 answer levels of the belief
             shift ||mu(S+(c,l)) - mu(S)|| -- the tree-style direct signal, computed by ACTUALLY folding
             every masked candidate (context-dependent; empty-context mean-shift dirs cannot substitute).
             The expected-Delta-NDCG variant of the spine needs an answer model / held labels at
             selection time (leakage), so the leakage-free mean-shift form of the brief's OR is used.
        (ii) SIGMA-GREEDY (argmax d_c' Sigma d_c) -- run alongside, reported, NOT required to win.
        (iii) RANDOM baseline.
      A = raw mu fold vs B = Sigma-gated fold mu' = mu' + Sigma(Sigma+S)^-1 (mu_new - mu') per ordering.
      PASS = gated monotone-nonneg AND SPINE > random by q4 (tail) AND full never below cold intercept.
      Raw-arm monotonicity is REPORTED (it attributes credit to the Sigma gate when raw is non-monotone)
      but is NOT a pass condition in either direction -- a raw arm that is already monotone is a strictly
      better outcome, not a failure. Success is NOT contingent on Sigma-greedy.

HARD RULES: NO data reduction anywhere (all answerer-train users in the fit; FULL val cohort in every gate);
FULL + TAIL always, scored z @ decoder.weight.T + decoder.BIAS (LEARNED bias -- NEVER popb); CPU only;
no LLM calls; 300-study users quarantined. DESIGN-ONLY: written, reviewed, NOT yet executed.

REVIEW ROUND 2 (all 3 blocking points addressed):
  #1 REAL sv threaded into every encoder fold that contains item evidence (analytic_state svs= param:
     fit uses e[1], G1a/G1c use u['ksv'][sel]); zero-sv path now HARD-ASSERTS concept-only/empty evidence
     (make_pb_example sets concept sv=0 -- 'value lives in the LEVEL' -- so G1b/G2 zero-sv is exact).
  #2 fit_alphas micro-chunks each batch by evidence size (gate_g1 pattern, Woodbury<=WOOD_MAX step 256 /
     dense step 32) with PER-CHUNK backward + gradient accumulation: live autograd memory bounded by one
     chunk, loss weighting identical to the unchunked mean, NO example dropped (HARD RULE #1).
  #3 one-time structural assert on the make_pb_example tuple (_assert_example_layout): dtypes, ranges,
     and the concept-level range check that catches a silent lv/kk transposition (kk is only 0..2).

INCREMENT REWRITE (this revision, DESIGN-ONLY, not yet executed): (a) fit objective replaced by the
increment objective above (precompute_increments + prefit_sign_proof + rewritten fit_alphas); (b)
SigmaOps.cross_user added (bilinear W Sigma D'); (c) gate_g2 runs SPINE + sigma-greedy + random side by
side, pass keyed on SPINE; (d) Delta_e cached at .cache/set_mn/inc_<base>.pt (save-all rule; cache
records n_users, the seed AND the objective-version string INC_TARGET_VER, all asserted on load so a
stale blob from the old subsample-target objective can never silently refit the wrong increments).
Frozen pbC load, G0 anchor, Woodbury Sigma, the --conc_dirs mean-shift override, G1a/b/c, canonical
z@Wd.T+bias scoring, FULL+TAIL all UNCHANGED.

REVIEW ROUND 3 (increment rewrite, all 4 blocking points addressed): #1 s* = s_i(mu_full), mu_full =
frozen fold of the FULL known half (all items at real sv/levels + the example's revealed concept
answers) -- term2 now carries signal, Delta_e properly signed; #2 sign-proof statistic Huber-clamped
(clamp(De,-beta,beta) * P^2 = exact negative loss gradient); #3 g2_pass no longer requires the raw arm
to be non-monotone (a_mono reported, never a veto); #4 Delta_e cache stores/asserts target=INC_TARGET_VER.
"""
import os, sys, time, argparse, math
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count()))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(os.cpu_count())

import set_mn as S
from set_mn import eval_student, make_batches, pad_batch, build_pb_users, make_pb_example, \
    load_answerer, concept_answers, sv_to_level
from signed_latent import load_arena_base, build_splits, cohort, ndcg10, safe_save, SEEDS, LO, log
import arena_core as AC

# REVIEWED base: train_varhead supplies frozen loading, r_pool encode, cohorts, spearman, freeze asserts.
# Importing it also runs S.set_grading("halfstar") (pbC_best is halfstar; NLEV=15), matching its header.
from train_varhead import load_frozen, encode_pool, build_val_recs, spearman, \
    trunk_snapshot, trunk_drift, empirical_sigma0, EXPECT_FULL, EXPECT_TAIL, OUT, D, META

NC = S.NC; NLEV = S.NLEV; REFUSE = S.LV_REFUSE; COFF = S.CLEVEL_OFFSET      # halfstar: 15 / 14 / 10
assert NLEV == 15 and REFUSE == 14 and COFF == 10, "expected halfstar grading (train_varhead sets it)"

WOOD_MAX = 256        # m <= WOOD_MAX -> Woodbury (m x m solve); m > WOOD_MAX -> dense 512 Lambda Cholesky
                      # (both EXACT; the switch only picks the cheaper exact factorization)
INC_SEED = 910_000    # per-user split seed (INC_SEED + user index) -> the Delta_e target is DETERMINISTIC
INC_TARGET_VER = "mu_full_v2"   # Delta_e objective version, stored+asserted in the cache blob (reviewer
                                # #4: a stale blob from the OLD subsample-target objective must not load)
TOK_BUDGET = 250_000  # leave-one-out fold chunking: rows_per_call * (m-1) <= budget. NON-LOSSY (HARD RULE
                      # #1): chunking only bounds one encoder call; every atom of every user is folded.
SPINE_CHUNK = 4096    # G2 SPINE candidate folds per encoder call (non-lossy chunking, ALL candidates kept)
N_BUCK = 5            # level buckets: 0=hated 1=meh 2=liked 3=loved 4=refuse
# halfstar ITEM levels 0..9 = half-stars 0.5..5.0 (sv_to_level: star = sv*2.25+2.75) -> ordinal bucket:
# 0.5-2.0 hated | 2.5-3.0 meh | 3.5-4.0 liked | 4.5-5.0 loved  (matches the answerer's ordinal thresholds)
ITEM_BUCKET = np.array([0, 0, 0, 0, 1, 1, 2, 2, 3, 3], np.int64)
_LAYOUT_OK = [False]  # one-time make_pb_example structural check (reviewer blocking #3)


# ============================= PrecAcc: 13 fitted scalars, nothing else =============================
class PrecAcc(nn.Module):
    """log_alpha (2 x 5): channel {0=item, 1=concept} x bucket {hated,meh,liked,loved,refuse}.
    v0 = s0 * (var_emp + vfloor): the anisotropic empirical prior VARIANCE, globally rescaled + floored
    (additive floor keeps everything differentiable; clamp would dead-zone the vfloor gradient).
    tau2 = observation-noise floor in the fit NLL (score-scale slack). exp-param => positivity, hence
    every atom is PSD precision => G1a/G1b hold BY CONSTRUCTION."""
    def __init__(self, var_emp):
        super().__init__()
        la = torch.zeros(2, N_BUCK)                     # alpha init 1.0 ...
        la[:, 4] = -4.0                                 # ... except refuse ~ 0.018 (near-zero information)
        self.log_alpha = nn.Parameter(la)
        self.log_s0 = nn.Parameter(torch.zeros(()))     # global prior-variance scale, init 1
        self.log_vfloor = nn.Parameter(torch.full((), math.log(1e-4)))   # additive variance floor
        self.log_tau2 = nn.Parameter(torch.full((), math.log(1e-2)))     # fit-NLL noise floor
        self.register_buffer("var_emp", torch.as_tensor(var_emp, dtype=torch.float32))

    def alphas(self):
        return torch.exp(self.log_alpha)                                 # (2, 5) > 0

    def v0(self):
        return torch.exp(self.log_s0) * (self.var_emp + torch.exp(self.log_vfloor))   # (512,) > 0

    def tau2(self):
        return torch.exp(self.log_tau2)


# ============================= concept directions (whitened member centroids) =============================
def build_dirs(enc, decoder, ni, members_path):
    """Unit atom directions. ITEMS: normalize(Wd[i]) rows (exact GGN/affinity directions -- NOT whitened;
    the decoder row IS the curvature direction of the multinomial GGN). CONCEPTS, two coherent modes:
    (a) members file supplied -> centroid_c = mean(Wd[members_c]) whitened with the reweight_geo2
        whitening (center by Wd.mean, strip Wd's top-1 PC -- the popularity component), normalized.
        Whitening is VALID here because the centroid lives in decoder (Wd) space.
    (b) FALLBACK (no members file) -> raw NORMALIZED Ec = the frozen encoder's concept token embedding,
        with NO Wd whitening: Ec lives in the encoder INPUT space, so centering by Wd.mean / stripping
        Wd's PC would be cross-space nonsense (near-parallel -m0-dominated atoms). This is varhead's own
        d_c convention. Returns (dirs_item, dirs_conc, dirs_conc_mode); the mode is saved in the blob so
        any G1b/G2 verdict is attributable to the direction set actually used."""
    Wd = decoder.weight.detach()                                          # (ni, 512)
    dirs_item = F.normalize(Wd, dim=-1)                                   # (ni, 512) unit rows
    m0 = Wd.mean(0)
    _, _, Vp = torch.pca_lowrank(Wd - m0, q=1, niter=8)                   # top-1 PC of centered Wd
    pc = F.normalize(Vp[:, 0], dim=0)                                     # (512,) popularity axis

    def whiten(X):                                                        # (N,512) -> centered, PC-stripped
        Xc = X - m0
        return Xc - torch.outer(Xc @ pc, pc)

    if members_path and os.path.exists(members_path):
        mm = np.load(members_path, allow_pickle=True)
        if "indptr" in getattr(mm, "files", []):                          # CSR: indptr (NC+1,), idx (nnz,)
            indptr, idx = mm["indptr"], mm["idx"]
            cent = torch.stack([Wd[idx[indptr[c]:indptr[c + 1]]].mean(0) if indptr[c + 1] > indptr[c]
                                else torch.zeros(D) for c in range(NC)])
        else:                                                             # object array of index arrays
            arr = mm["members"] if "members" in getattr(mm, "files", []) else mm
            cent = torch.stack([Wd[np.asarray(a, np.int64)].mean(0) if len(a) else torch.zeros(D)
                                for a in arr])
        log(f"[dirs] concept centroids from members file {members_path}")
        empty = int((cent.norm(dim=-1) < 1e-6).sum())
        assert empty == 0, f"build_dirs: {empty} concepts have NO members (whiten(0) would be a bogus -m0 atom)"
        dirs_conc = F.normalize(whiten(cent), dim=-1)                     # (NC, 512) unit rows, Wd-space
        mode = "whitened_member_centroid"
    else:
        if members_path:                                                  # path GIVEN but missing = hard fail
            raise FileNotFoundError(f"--members file not found: {members_path} (refusing silent fallback)")
        Ec = enc.item_emb.weight[ni:ni + NC].detach().clone()             # encoder-INPUT space tokens
        dirs_conc = F.normalize(Ec, dim=-1)                               # RAW normalized -- NO Wd whitening
        mode = "raw_Ec_no_whiten"
        log("[dirs] *** NO members file: FALLBACK d_c = raw normalized Ec (encoder-input space; Wd "
            "whitening deliberately NOT applied -- cross-space). Pass --members for the exact "
            "belief-pool whitened decoder-space d_c ***")
    log(f"[dirs] dirs_conc_mode={mode}")
    return dirs_item, dirs_conc, mode


# ============================= evidence -> precision atoms U =============================
def token_bucket(tid, lv, ni):
    """(channel, bucket) for one token. Items: halfstar level 0..9 -> ordinal bucket; concepts:
    token level COFF+vl -> vl; REFUSE -> bucket 4 (either channel; only concepts refuse in practice)."""
    if lv == REFUSE:
        return (1 if tid >= ni else 0), 4
    if tid >= ni:
        return 1, int(np.clip(lv - COFF, 0, 3))
    return 0, int(ITEM_BUCKET[np.clip(lv, 0, 9)])


def build_U(seqs, model, dirs_item, dirs_conc, ni, alpha_override=None):
    """seqs = list of (ids_np, lv_np). Returns U (B, 512, mmax): column j = sqrt(alpha_j) d_j, zero-padded
    (zero columns are inert: M = I + U'V0U stays SPD, their contribution is exactly 0). Differentiable
    in model.log_alpha through the sqrt(alpha) gather. alpha_override: explicit (2, N_BUCK) tensor used
    instead of model.alphas() -- the pre-fit sign proof passes the init alphas with the CONCEPT row
    zeroed (sqrt(0)=0 -> inert columns -> Sigma(S) == Sigma(S\\c) exactly for concept atoms)."""
    B = len(seqs); mmax = max(1, max(len(s[0]) for s in seqs))
    a = model.alphas() if alpha_override is None else alpha_override
    U = torch.zeros(B, D, mmax)
    cols = []
    for r, (tid, tlv) in enumerate(seqs):
        for j in range(len(tid)):
            ch, bk = token_bucket(int(tid[j]), int(tlv[j]), ni)
            dvec = dirs_conc[int(tid[j]) - ni] if int(tid[j]) >= ni else dirs_item[int(tid[j])]
            cols.append((r, j, ch, bk, dvec))
    if cols:
        rr = torch.tensor([c[0] for c in cols]); jj = torch.tensor([c[1] for c in cols])
        chb = torch.tensor([c[2] for c in cols]); bkb = torch.tensor([c[3] for c in cols])
        Dm = torch.stack([c[4] for c in cols])                            # (n, 512)
        Uc = torch.sqrt(a[chb, bkb]).unsqueeze(-1) * Dm                   # (n, 512) grads -> log_alpha
        U = torch.zeros(B, D, mmax, dtype=Uc.dtype)
        U[rr, :, jj] = Uc                                                 # scatter columns (advanced indexing)
    return U


# ============================= exact Sigma algebra (Woodbury / dense switch) =============================
class SigmaOps:
    """Exact Sigma = (Lambda0 + U U^T)^{-1} operations for one padded batch. Woodbury for m <= WOOD_MAX
    (m x m Cholesky), dense 512 Lambda Cholesky otherwise. All methods differentiable."""
    def __init__(self, U, v0):
        self.U = U; self.v0 = v0                                          # (B,d,m), (d,)
        self.B, self.d, self.m = U.shape
        self.dense = self.m > WOOD_MAX
        if self.dense:
            Lam = torch.diag(1.0 / v0).unsqueeze(0) + torch.einsum("bdm,bem->bde", U, U)
            self.Lc = torch.linalg.cholesky(Lam)                          # (B,d,d)
        else:
            self.Uv = U * v0.view(1, -1, 1)                               # V0 U
            M = torch.eye(self.m).unsqueeze(0) + torch.einsum("bdm,bdk->bmk", U, self.Uv)
            self.Mc = torch.linalg.cholesky(M)                            # (B,m,m) SPD (zero cols inert)

    def quad(self, W):
        """w' Sigma w for a SHARED bank W (N,d) -> (B,N)."""
        if self.dense:
            Y = torch.cholesky_solve(W.T.unsqueeze(0).expand(self.B, -1, -1), self.Lc)   # (B,d,N)
            return torch.einsum("nd,bdn->bn", W, Y)
        q0 = (W ** 2) @ self.v0                                           # (N,) prior part
        T = torch.einsum("nd,bdm->bnm", W, self.Uv)                       # U^T V0 w
        sol = torch.cholesky_solve(T.transpose(1, 2), self.Mc)            # (B,m,N)
        return q0.unsqueeze(0) - (T * sol.transpose(1, 2)).sum(-1)

    def quad_user(self, r, W):
        """w' Sigma w for user r with its OWN bank W (n,d) -> (n,)."""
        if self.dense:
            Y = torch.cholesky_solve(W.T, self.Lc[r])
            return torch.einsum("nd,dn->n", W, Y)
        q0 = (W ** 2) @ self.v0
        T = W @ self.Uv[r]                                                # (n,m)
        sol = torch.cholesky_solve(T.T, self.Mc[r])                       # (m,n)
        return q0 - (T * sol.T).sum(-1)

    def cross_user(self, r, W, Dm):
        """Bilinear W Sigma D^T for user r: W (n,d) probe rows, Dm (m,d) atom directions -> (n,m).
        Woodbury: W Sigma D' = W V0 D' - (W V0 U) M^{-1} (U' V0 D'). Same factorization as quad_user
        (quad_user(r, X) == diag of cross_user(r, X, X)); differentiable."""
        if self.dense:
            Y = torch.cholesky_solve(Dm.T, self.Lc[r])                    # (d,m) = Sigma D'
            return W @ Y
        base = (W * self.v0) @ Dm.T                                       # W V0 D'          (n,m)
        TW = W @ self.Uv[r]                                               # W V0 U           (n,mpad)
        TD = Dm @ self.Uv[r]                                              # D V0 U           (m,mpad)
        sol = torch.cholesky_solve(TD.T, self.Mc[r])                      # M^{-1} U'V0 D'   (mpad,m)
        return base - TW @ sol

    def trace(self):
        """tr(Sigma) -> (B,)."""
        if self.dense:
            Sig = torch.cholesky_inverse(self.Lc)                         # (B,d,d) exact inverse
            return Sig.diagonal(dim1=1, dim2=2).sum(-1)
        G = torch.einsum("bdm,bdk->bmk", self.Uv, self.Uv)                # U^T V0^2 U
        sol = torch.cholesky_solve(G, self.Mc)                            # M^{-1} G
        return self.v0.sum() - sol.diagonal(dim1=1, dim2=2).sum(-1)

    def gate(self, mu_prev, mu_new, Sg):
        """Sigma-gated fold mu' = mu_prev + Sigma (Sigma + Sg I)^{-1} (mu_new - mu_prev).
        Identity: Sigma(Sigma + Sg I)^{-1} = (I + Sg Lambda)^{-1} = (diag(1 + Sg/v0) + Sg U U^T)^{-1}."""
        x = mu_new - mu_prev                                              # (B,d)
        g = 1.0 + Sg / self.v0                                            # (d,)
        if self.dense:
            A = torch.diag(g).unsqueeze(0) + Sg * torch.einsum("bdm,bem->bde", self.U, self.U)
            y = torch.linalg.solve(A, x.unsqueeze(-1)).squeeze(-1)
            return mu_prev + y
        xg = x / g                                                        # diag(g)^{-1} x
        Ug = self.U / g.view(1, -1, 1)                                    # diag(g)^{-1} U
        Mi = torch.eye(self.m).unsqueeze(0) / Sg + torch.einsum("bdm,bdk->bmk", self.U, Ug)
        sol = torch.linalg.solve(Mi, torch.einsum("bdm,bd->bm", self.U, xg).unsqueeze(-1)).squeeze(-1)
        y = xg - torch.einsum("bdm,bm->bd", Ug, sol)
        return mu_prev + y


def analytic_state(enc, model, seqs, dirs_item, dirs_conc, ni, grad=False, svs=None):
    """One frozen forward + analytic Sigma: seqs -> (mu (B,512), SigmaOps). mu is the frozen pbC point
    embedding -- NOTHING about the mean path changes. Sigma is pure function of the evidence atoms.
    svs: per-seq float scalar-value arrays for ITEM evidence, threaded into the encoder's vals input so
    the folded mu matches the eval_student / make_pb_example convention EXACTLY (reviewer blocking #1:
    zero vals would silently corrupt every folded mu if the encoder consumes vals, and G0 -- which uses
    eval_student with REAL vals -- could never catch it). svs=None (vals=0) is ONLY valid for empty or
    concept-only evidence: make_pb_example sets sv=0 for concept tokens ('value lives in the LEVEL')."""
    B = len(seqs); L = max(1, max(len(s[0]) for s in seqs))
    ids = np.zeros((B, L), np.int64); lv = np.zeros((B, L), np.int64); pad = np.ones((B, L), bool)
    vals = np.zeros((B, L), np.float32)
    for r, (tid, tlv) in enumerate(seqs):
        n = len(tid)
        if n:
            ids[r, :n] = tid; lv[r, :n] = tlv; pad[r, :n] = False
            if svs is not None:
                assert len(svs[r]) == n, "svs must align 1:1 with seqs"
                vals[r, :n] = svs[r]
    if svs is None:
        # zero-vals path: assert the convention that justifies it (no item token may pass through here)
        assert all((np.asarray(s[0]) >= ni).all() for s in seqs), \
            "item evidence folded without real sv -- thread svs= (reviewer blocking #1)"
    mu, _ = encode_pool(enc, torch.from_numpy(ids), torch.from_numpy(vals),
                        torch.from_numpy(pad), torch.from_numpy(lv))      # frozen, no grad
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        U = build_U(seqs, model, dirs_item, dirs_conc, ni)
        ops = SigmaOps(U, model.v0())
    return mu, ops


def fold_mu(enc, seqs, svs, ni):
    """Frozen mu-only fold (no Sigma, no model): seqs = [(tid_np, lv_np)], svs = aligned float sv arrays
    (REAL item sv; concept sv = 0 by the make_pb_example convention). Used by the Delta_e precompute
    (full-S and leave-one-out folds) and by the G2 SPINE candidate folds (concept-only, sv=0)."""
    B = len(seqs); L = max(1, max(len(s[0]) for s in seqs))
    ids = np.zeros((B, L), np.int64); lv = np.zeros((B, L), np.int64)
    pad = np.ones((B, L), bool); vals = np.zeros((B, L), np.float32)
    for r, (tid, tlv) in enumerate(seqs):
        n = len(tid)
        if n:
            assert len(svs[r]) == n, "fold_mu: svs must align 1:1 with seqs"
            ids[r, :n] = tid; lv[r, :n] = tlv; pad[r, :n] = False; vals[r, :n] = svs[r]
    with torch.no_grad():
        mu, _ = encode_pool(enc, torch.from_numpy(ids), torch.from_numpy(vals),
                            torch.from_numpy(pad), torch.from_numpy(lv))
    return mu


def atom_dirs(tid, dirs_item, dirs_conc, ni):
    """(m,) token ids -> (m, 512) unit atom directions (item rows / concept rows)."""
    t = torch.from_numpy(np.asarray(tid, np.int64))
    Dm = torch.empty(len(tid), D, dtype=dirs_item.dtype)
    conc = t >= ni
    if conc.any():
        Dm[conc] = dirs_conc[t[conc] - ni]
    if (~conc).any():
        Dm[~conc] = dirs_item[t[~conc]]
    return Dm


# ============================= ALPHA-FIT (increment objective; Delta_e precompute + sign proof) =======
def _assert_example_layout(e, ni):
    """One-time structural check of the make_pb_example tuple (reviewer blocking #3): the fit assumes
    e[0]=token ids, e[1]=float sv, e[2]=levels, e[4]=held-liked targets positionally; lv (e[2]) and
    kk (e[3]) are both small ints, so a transposition would corrupt the fit SILENTLY. The concept-level
    range check ([COFF, COFF+3] or REFUSE, i.e. >= 10) catches an lv/kk swap (kk is only ever 0..2)."""
    assert len(e) >= 5, f"make_pb_example returned {len(e)}-tuple, expected (tid, sv, lv, kk, tgt)"
    tid = np.asarray(e[0]); sv = np.asarray(e[1]); lv = np.asarray(e[2]); tg = np.asarray(e[4])
    assert tid.dtype.kind in "iu" and lv.dtype.kind in "iu" and tg.dtype.kind in "iu", \
        f"integer fields expected at e[0]/e[2]/e[4], got {tid.dtype}/{lv.dtype}/{tg.dtype}"
    assert sv.dtype.kind == "f", f"e[1] must be float sv, got {sv.dtype} (layout transposed?)"
    assert len(tid) == len(sv) == len(lv), "tid/sv/lv length mismatch"
    assert (tid >= 0).all() and (tid < ni + NC).all(), "token id out of [0, ni+NC)"
    assert (lv >= 0).all() and (lv < NLEV).all(), "level out of [0, NLEV)"
    assert (tg >= 0).all() and (tg < ni).all(), "target out of item range [0, ni)"
    conc = tid >= ni
    if conc.any():
        clv = lv[conc]
        assert (((clv >= COFF) & (clv <= COFF + 3)) | (clv == REFUSE)).all(), \
            "concept token levels must be in [COFF, COFF+3] or REFUSE -- lv/kk transposition?"
        assert np.abs(sv[conc]).max() == 0.0, "concept sv must be 0 (value lives in the LEVEL)"
    if (~conc).any():
        assert (lv[~conc] < 10).all(), "item halfstar levels must be 0..9 -- lv/kk transposition?"
    log("[fit] make_pb_example layout check PASS (tid, sv, lv, kk, tgt)")


def precompute_increments(enc, decoder, users, KT, VT, ni, path, max_atoms=0):
    """DELTA_E PRECOMPUTE (frozen, NO alpha, ONCE, cached -- save-all rule). Per train user (ALL of them,
    HARD RULE #1; determinstic split rng = INC_SEED + user index so the target never moves between runs):
      make_pb_example -> evidence set S = (tid, sv, lv) (a SUB-interview: heavy item dropout, 1-32
      concepts) + held-liked probe items tg (NEVER folded into S or any S\\a);
      TARGET s*_i = <mu_full, Wd_i> + bias_i, mu_full = frozen fold of the user's FULL KNOWN HALF =
      ALL of u['items'] at their REAL sv/levels + the example's revealed concept answers (reviewer
      blocking #1: with s* = s_S the second term collapses to 0 and Delta_e >= 0 identically -- the
      objective loses the ability to assign zero/negative credit to uninformative atoms);
      for every atom a in S: leave-one-out fold mu(S\\a) ->
          De[i, a] = (s*_i - s_i(mu(S\\a)))^2 - (s*_i - s_i(mu(S)))^2      [SIGNED]
    COST = 2 + |S| encoder folds per user (mu_full + mu(S) + the LOO folds; batched under TOK_BUDGET --
    chunking is NON-LOSSY, every atom/user/probe kept). Returns a list aligned with `users`
    (None where make_pb_example yields no usable example); each entry stores tid/sv/lv/tg/ch/bk/De."""
    if os.path.exists(path):
        blob = torch.load(path, map_location="cpu")
        assert blob["n_users"] == len(users) and blob["seed"] == INC_SEED, \
            f"increment cache {path} was built for a different cohort/seed -- delete it or fix the cohort"
        assert blob.get("target") == INC_TARGET_VER, \
            f"increment cache {path} has objective version {blob.get('target')!r} != {INC_TARGET_VER!r} " \
            "(stale blob from the old subsample-target objective) -- delete it and re-precompute"
        log(f"[inc] loaded Delta_e cache {path} ({sum(1 for e in blob['examples'] if e is not None)} users)")
        return blob["examples"]
    Wd = decoder.weight.detach(); bd = decoder.bias.detach()
    exs = [make_pb_example(u, KT, VT, np.random.default_rng(INC_SEED + ui)) for ui, u in enumerate(users)]
    ms = [len(e[0]) for e in exs if e is not None]
    log(f"[inc] precompute over {len(ms)}/{len(users)} usable users | atoms {sum(ms):,} | "
        f"LOO encoder tokens ~{sum(m * (m - 1) for m in ms):,} (ONE-TIME, cached to {path})")
    out = []; t0 = time.time()
    for ui, e in enumerate(exs):
        if e is None:
            out.append(None); continue
        if not _LAYOUT_OK[0]:
            _assert_example_layout(e, ni); _LAYOUT_OK[0] = True
        u = users[ui]                                                     # exs is enumerate(users)-aligned
        tid = np.asarray(e[0], np.int64); sv = np.asarray(e[1], np.float32)
        lv = np.asarray(e[2], np.int64); tg = np.asarray(e[4], np.int64)
        m = len(tid)
        Wt = Wd[torch.from_numpy(tg)]; bt = bd[torch.from_numpy(tg)]
        # TARGET fold (reviewer #1): FULL KNOWN HALF = ALL items at REAL sv/levels + the example's
        # revealed concept answers (concept sv = 0 by convention -- value lives in the LEVEL).
        cm = tid >= ni                                                    # concept atoms of the example
        full_tid = np.concatenate([u["items"].astype(np.int64), tid[cm]])
        full_sv = np.concatenate([u["sv"].astype(np.float32), np.zeros(int(cm.sum()), np.float32)])
        full_lv = np.concatenate([np.asarray(sv_to_level(u["sv"]), np.int64), lv[cm]])
        mus2 = fold_mu(enc, [(tid, lv), (full_tid, full_lv)], [sv, full_sv], ni)
        mu_S, mu_full = mus2[0], mus2[1]                                  # frozen full-S fold, target fold
        s_S = mu_S @ Wt.T + bt                                            # s_i(mu(S))
        s_star = mu_full @ Wt.T + bt                                      # TARGET s*_i = s_i(mu_full)
        term2 = (s_star - s_S) ** 2                                       # > 0 in general -> De is SIGNED
        # LOO-evaluated atoms: cap per-user to max_atoms (folds mu(S\a) STILL use the full S -- we only
        # compute De for a deterministic subset of atoms a; each atom is an independent (channel,level)
        # training triple, so a subset gives unbiased bucket coverage and tames the O(m^2) tail).
        if max_atoms and m > max_atoms:
            ea = np.sort(np.random.default_rng(INC_SEED * 7 + ui).choice(m, max_atoms, replace=False))
        else:
            ea = np.arange(m)
        ne = len(ea)
        De = np.empty((len(tg), ne), np.float32)
        rows = max(1, TOK_BUDGET // max(m - 1, 1))                        # non-lossy LOO chunking
        for a0 in range(0, ne, rows):
            aa = ea[a0:a0 + rows]
            loo = fold_mu(enc, [(np.delete(tid, a), np.delete(lv, a)) for a in aa],
                          [np.delete(sv, a) for a in aa], ni)             # (rows, 512)
            s_loo = loo @ Wt.T + bt.unsqueeze(0)                          # (rows, n)
            De[:, a0:a0 + len(aa)] = ((s_star.unsqueeze(0) - s_loo) ** 2 - term2.unsqueeze(0)).T.numpy()
        ch = np.empty(ne, np.int64); bk = np.empty(ne, np.int64)
        for j in range(ne):
            ch[j], bk[j] = token_bucket(int(tid[ea[j]]), int(lv[ea[j]]), ni)
        out.append(dict(tid=tid[ea], sv=sv[ea], lv=lv[ea], tg=tg, ch=ch, bk=bk, De=De))
        if (ui + 1) % 1000 == 0:
            el = (time.time() - t0) / 60
            log(f"  [inc] {ui + 1}/{len(users)} users ({el:.1f}m, ~{el / (ui + 1) * len(users):.0f}m total)")
    safe_save({"examples": out, "n_users": len(users), "seed": INC_SEED, "target": INC_TARGET_VER}, path)
    allDe = np.concatenate([o["De"].ravel() for o in out if o is not None])
    log(f"[inc] Delta_e precompute done ({(time.time() - t0) / 60:.1f}m) -> {path} | "
        f"De signed: {float((allDe > 0).mean()):.1%} pos / {float((allDe < 0).mean()):.1%} neg, "
        f"mean {float(allDe.mean()):.4g} (a ~100% pos split would mean the old degenerate target)")
    return out


def _size_chunks(sizes):
    """(Woodbury small / dense big) index chunks -- the shared gate_g1/fit chunking pattern."""
    big = sizes > WOOD_MAX
    for idx in (np.flatnonzero(~big), np.flatnonzero(big)):
        if len(idx) == 0:
            continue
        step = 256 if not big[idx[0]] else 32
        for c0 in range(0, len(idx), step):
            yield idx[c0:c0 + step]


def prefit_sign_proof(model, decoder, inc, dirs_item, dirs_conc, ni, huber_beta):
    """MANDATORY GATE (seconds, BEFORE the fit). At alpha_conc = 0 the concept-channel gradient of the
    increment loss is dL/dalpha_conc < 0  iff
        G = sum_{concept triples} clamp(De[u,i,c], -beta, beta) * (w_i' Sigma(S\\c) d_c)^2  >  0.
    The clamp is EXACT (reviewer blocking #2): at alpha_conc = 0 the residual is r = Delta_v - De = -De,
    and smooth-L1' (r) = r/beta for |r| < beta, sign(r) outside -- so the per-triple gradient weight is
    -clamp(De/beta, -1, 1), and with dDelta_v/dalpha|_0 = (w'Sigma d_c)^2,
        dL/dalpha_conc = -(1/beta) * G  (a positive constant times -G; sign identical).
    De is SIGNED under the mu_full target, so the unclamped sum is NOT the gradient sign -- large
    |De| > beta triples must saturate exactly as the Huber loss saturates them. The concept
    row of alpha is zeroed EXACTLY, so concept atoms add no precision and Sigma(S) == Sigma(S\\c) -- one
    SigmaOps per chunk yields every LOO cross-term. Item alphas / Lambda0 sit at their init values.
    G > 0 guarantees alpha_conc leaves zero (the decoupling trap is structurally gone). G <= 0 => bug or
    data problem: STOP, do not train."""
    Wd = decoder.weight.detach()
    a0 = model.alphas().detach().clone(); a0[1, :] = 0.0                  # concept channel EXACTLY zero
    ex = [e for e in inc if e is not None and (e["ch"] == 1).any()]
    assert ex, "sign proof: no train example contains a concept atom -- data problem, cannot certify"
    Gtot = 0.0; Gbk = np.zeros(N_BUCK); ntrip = 0
    sizes = np.array([len(e["tid"]) for e in ex])
    t0 = time.time()
    with torch.no_grad():
        for sub in _size_chunks(sizes):
            ce = [ex[i] for i in sub]
            seqs = [(e["tid"], e["lv"]) for e in ce]
            U = build_U(seqs, model, dirs_item, dirs_conc, ni, alpha_override=a0)
            ops = SigmaOps(U, model.v0().detach())
            for r, e in enumerate(ce):
                cc = np.flatnonzero(e["ch"] == 1)                         # concept atom positions in S
                Dm = dirs_conc[torch.from_numpy(e["tid"][cc] - ni)]       # (mc, 512)
                P = ops.cross_user(r, Wd[torch.from_numpy(e["tg"])], Dm)  # (n, mc) = w' Sigma(S\c) d_c
                Dcl = torch.from_numpy(e["De"][:, cc]).clamp(-huber_beta, huber_beta)  # Huber gradient weight
                contrib = (Dcl * P ** 2).sum(0).numpy()                   # (mc,)
                Gtot += float(contrib.sum()); ntrip += e["De"].shape[0] * len(cc)
                np.add.at(Gbk, e["bk"][cc], contrib)
    log(f"[sign-proof] G = {Gtot:.6g} (Huber-clamped, beta={huber_beta}) over {ntrip:,} concept triples "
        f"({(time.time() - t0):.0f}s) | per-bucket {np.round(Gbk, 4).tolist()} (hated/meh/liked/loved/refuse)")
    assert Gtot > 0, f"SIGN PROOF FAILED: G = {Gtot:.6g} <= 0 -- alpha_conc would stay at zero. " \
                     "Bug or data problem (dead concept dirs / all-zero De?): STOP, do not train."
    return {"G": float(Gtot), "G_bucket": Gbk.tolist(), "n_triples": int(ntrip),
            "huber_beta": float(huber_beta)}


def fit_alphas(model, decoder, inc, dirs_item, dirs_conc, ni, args):
    """INCREMENT-OBJECTIVE fit (replaces the absolute residual-NLL that zeroed alpha_conc). Per user:
    ONE differentiable SigmaOps on the full evidence S, then for every atom a the EXACT downdate identity
        Delta_v_a = alpha_a * (w' Sigma(S) d_a)^2 / (1 - alpha_a * d_a' Sigma(S) d_a)
    predicts the LOO variance-reduction; L = mean_u mean_{i,a} Huber_beta(Delta_v - Delta_e). Delta_e is
    precomputed and FROZEN, so this loop touches NO encoder. The denominator is in (0,1] mathematically
    (Lambda(S\\a) SPD); clamped at 1e-6 for float safety only and the clamp rate is monitored. tau2 is
    INERT (no gradient reaches it). ALL usable train users every epoch (HARD RULE #1); eval users never
    enter. Chunked backward per batch (reviewer #2 pattern) keeps autograd memory bounded."""
    Wd = decoder.weight.detach()
    ex = [e for e in inc if e is not None]
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    order = np.argsort([len(e["tid"]) for e in ex])                       # size-sorted -> tight chunks
    B0 = 128
    batches_all = [order[i:i + B0] for i in range(0, len(order), B0)]
    log(f"[fit] INCREMENT objective | {len(ex)} users, {len(batches_all)} batches/epoch x {args.epochs} "
        f"epochs | huber beta {args.huber} | lr {args.lr} (deterministic targets: same increments each epoch)")
    hist = []
    for ep in range(args.epochs):
        rng = np.random.default_rng(500 + ep)
        batches = list(batches_all); rng.shuffle(batches)
        t0 = time.time(); run = 0.0; run_dv = 0.0; run_de = 0.0; nb = 0; nclamp = 0; natom = 0
        for bat in batches:
            be = [ex[i] for i in bat]
            sizes = np.array([len(e["tid"]) for e in be]); n_ex = len(be)
            mon_dv = []; mon_de = []; loss_tot = 0.0
            opt.zero_grad()
            for sub in _size_chunks(sizes):
                ce = [be[i] for i in sub]
                seqs = [(e["tid"], e["lv"]) for e in ce]
                with torch.enable_grad():
                    U = build_U(seqs, model, dirs_item, dirs_conc, ni)    # grads -> log_alpha
                    ops = SigmaOps(U, model.v0())                         # grads -> s0, vfloor
                    a = model.alphas()
                    closses = []
                    for r, e in enumerate(ce):
                        Dm = atom_dirs(e["tid"], dirs_item, dirs_conc, ni)
                        alp = a[torch.from_numpy(e["ch"]), torch.from_numpy(e["bk"])]   # (m,)
                        q = ops.quad_user(r, Dm)                          # d_a' Sigma(S) d_a   (m,)
                        P = ops.cross_user(r, Wd[torch.from_numpy(e["tg"])], Dm)        # (n,m)
                        rden = 1.0 - alp * q                              # in (0,1] mathematically
                        nclamp += int((rden.detach() < 1e-6).sum()); natom += len(alp)
                        dv = alp.unsqueeze(0) * P ** 2 / rden.clamp_min(1e-6).unsqueeze(0)
                        De = torch.from_numpy(e["De"])
                        closses.append(F.smooth_l1_loss(dv, De, beta=args.huber, reduction="mean"))
                        mon_dv.append(float(dv.detach().mean())); mon_de.append(float(De.mean()))
                    closs = torch.stack(closses).sum() / n_ex             # partial mean -> grads accumulate
                    closs.backward()
                    loss_tot += float(closs)
            opt.step()
            run += loss_tot; run_dv += float(np.mean(mon_dv)); run_de += float(np.mean(mon_de)); nb += 1
            if nb % 50 == 0:
                aa_ = model.alphas().detach().numpy()
                log(f"  [fit] ep{ep} b{nb}/{len(batches)} L={run / nb:.5f} "
                    f"dv={run_dv / nb:.4g} de={run_de / nb:.4g} clamp={nclamp}/{natom} "
                    f"s0={float(torch.exp(model.log_s0)):.3f} "
                    f"a_item={np.round(aa_[0], 3).tolist()} a_conc={np.round(aa_[1], 3).tolist()} "
                    f"({(time.time() - t0) / 60:.1f}m)")
        hist.append(dict(ep=ep, loss=run / max(nb, 1), dv=run_dv / max(nb, 1), de=run_de / max(nb, 1),
                         clamp_frac=nclamp / max(natom, 1)))
        log(f"[fit ep{ep + 1}] L={run / max(nb, 1):.5f} clamp={nclamp}/{natom} "
            f"({(time.time() - t0) / 60:.1f}m)")
    aa_ = model.alphas().detach().numpy()
    log(f"[fit] FINAL alpha item={np.round(aa_[0], 4).tolist()} concept={np.round(aa_[1], 4).tolist()} "
        f"s0={float(torch.exp(model.log_s0)):.4f} vfloor={float(torch.exp(model.log_vfloor)):.4g} "
        f"(tau2 inert at init {float(model.tau2()):.4g})")
    if hist and hist[-1]["clamp_frac"] > 1e-3:
        log("[fit] *** WARNING: downdate denominator clamped on >0.1% of atoms -- alpha pushing the "
            "one-atom precision toward the whole posterior; inspect before trusting Sigma ***")
    if float(np.max(aa_[1, :4])) < 0.02:
        log("[fit] *** WARNING: alpha_conc collapsed despite the sign proof -- the Huber tail or the "
            "shared Lambda0 fit is fighting the concept channel; report, do not paper over ***")
    return hist


# ============================= G1: posterior usable (analytic) =============================
def gate_g1(enc, model, decoder, recs, ni, dirs_item, dirs_conc, rng_seed=7):
    Wd = decoder.weight.detach(); bd = decoder.bias.detach()
    rng = np.random.default_rng(rng_seed)
    res = {}
    # ---------- (a) tr(Sigma) over NESTED evidence prefixes (guaranteed monotone => assert) ----------
    ks = [0, 1, 2, 4, 8, 16, 32, "full"]
    perms = [rng.permutation(len(u["ki"])) for u in recs]                 # ONE permutation/user -> prefixes
    traces = {}
    for k in ks:
        acc = []
        for b in range(0, len(recs), 256):
            ch = recs[b:b + 256]
            seqs = []; svs = []                                           # REAL item sv threaded (blocking #1)
            for r, u in enumerate(ch):
                n = len(u["ki"]) if k == "full" else min(int(k), len(u["ki"]))
                sel = perms[b + r][:n]
                seqs.append((u["ki"][sel], sv_to_level(u["ksv"][sel]))); svs.append(u["ksv"][sel])
            # chunk by evidence size so dense path stays memory-sane (NON-LOSSY: every user still scored)
            sizes = np.array([len(s[0]) for s in seqs])
            big = sizes > WOOD_MAX
            for idx in (np.flatnonzero(~big), np.flatnonzero(big)):
                if len(idx) == 0:
                    continue
                step = 256 if not big[idx[0]] else 32
                for c0 in range(0, len(idx), step):
                    ss = idx[c0:c0 + step]
                    _, ops = analytic_state(enc, model, [seqs[i] for i in ss], dirs_item, dirs_conc,
                                            ni, svs=[svs[i] for i in ss])
                    acc.append(ops.trace().numpy())
        traces[str(k)] = float(np.mean(np.concatenate(acc)))
    vals_ = [traces[str(k)] for k in ks]
    g1a = all(vals_[i] > vals_[i + 1] for i in range(len(vals_) - 1))
    res["g1a_traces"] = {str(k): round(traces[str(k)], 4) for k in ks}; res["g1a_pass"] = bool(g1a)
    log(f"[G1a] tr(Sigma) by k (nested): {res['g1a_traces']} strictly-decreasing={'PASS' if g1a else 'FAIL'}")
    assert g1a, "G1a VIOLATED -- by-construction property failed => CODE BUG in SigmaOps/build_U, fix it"

    # ---------- (b) DIRECTIONAL: fold concept c -> relative drop along d_c vs other concepts ----------
    with torch.no_grad():
        _, ops0 = analytic_state(enc, model, [(np.zeros(0, np.int64), np.zeros(0, np.int64))],
                                 dirs_item, dirs_conc, ni)
        q0_all = ops0.quad(dirs_conc)[0].clamp_min(1e-12)                 # (NC,) prior var along each d_c
        rel_own = np.zeros(NC); rel_oth = np.zeros(NC)
        for b in range(0, NC, 256):
            cids = np.arange(b, min(b + 256, NC))
            seqs = [(np.array([ni + c], np.int64), np.array([COFF + 3], np.int64)) for c in cids]  # "loved"
            _, ops1 = analytic_state(enc, model, seqs, dirs_item, dirs_conc, ni)
            q1 = ops1.quad(dirs_conc)                                     # (B, NC)
            R = ((q0_all.unsqueeze(0) - q1) / q0_all.unsqueeze(0)).numpy()
            diag = R[np.arange(len(cids)), cids]
            rel_own[cids] = diag
            rel_oth[cids] = (R.sum(1) - diag) / max(NC - 1, 1)
    mo, mr = float(rel_own.mean()), float(rel_oth.mean())
    ratio = mo / max(mr, 1e-12) if mr > 0 else (float("inf") if mo > 0 else 0.0)
    res["g1b_rel_own"] = round(mo, 6); res["g1b_rel_other"] = round(mr, 6)
    res["g1b_ratio"] = round(ratio, 3); res["g1b_pass"] = bool(mo > 0 and ratio >= 2.0)
    log(f"[G1b] directional (ALL {NC} concepts): own {mo:.3g} vs other {mr:.3g} ratio {ratio:.2f} "
        f"({'PASS >=2x' if res['g1b_pass'] else 'FAIL'})")
    assert res["g1b_pass"], "G1b VIOLATED -- precision added along d_c must shrink d_c most => CODE BUG " \
                            "(check dirs_conc normalization / alpha[concept,loved] > 0)"

    # ---------- (c) DECISIVE calibration: sqrt(w'Sigma w) vs realized held-item NLL at mu ----------
    unc = []; err = []
    with torch.no_grad():
        for b in range(0, len(recs), 256):
            ch = recs[b:b + 256]
            seqs = []; svs = []                                           # REAL item sv threaded (blocking #1)
            for u in ch:
                n = min(8, len(u["ki"]))
                sel = rng.choice(len(u["ki"]), size=n, replace=False)
                seqs.append((u["ki"][sel], sv_to_level(u["ksv"][sel]))); svs.append(u["ksv"][sel])
            mu, ops = analytic_state(enc, model, seqs, dirs_item, dirs_conc, ni, svs=svs)
            logsm = F.log_softmax(mu @ Wd.T + bd, -1)
            for r, u in enumerate(ch):
                w = Wd[u["hl"]].mean(0)
                w = (w / w.norm().clamp_min(1e-8)).unsqueeze(0)           # SAME normalized probe as varhead
                unc.append(float(torch.sqrt(ops.quad_user(r, w)[0])))
                err.append(float(-logsm[r, u["hl"]].mean()))
    rho = spearman(unc, err)
    res["g1c_rho"] = round(rho, 4); res["g1c_pass"] = bool(rho >= 0.25)   # DECISIVE bar: beat head's 0.219
    log(f"[G1c] calibration rho(sqrt(w'Sw), held NLL) = {rho:.4f} over {len(unc)} users "
        f"({'PASS >=0.25' if res['g1c_pass'] else 'FAIL (learned head scored 0.219 -- must beat it)'})")
    return res


# ============================= G2: decisive cold-start curves (analytic Sigma) ========================
def gate_g2(enc, model, decoder, base, recs, KV, VV, ni, dirs_item, dirs_conc, Sgates, q_max=8, rng_seed=11):
    """Identical harness to train_varhead.gate_g2 (same answerability-masked pool for ALL orderings, same
    refusal-burns-turn rule, same scoring) with THREE selection orderings run side by side:
      spine  = DIRECT informativeness (the decisive arm): value(c) = mean over the 4 answer levels of
               ||mu(S + (c, l)) - mu(S)|| -- expected belief movement under a uniform answer model, the
               tree-style direct signal, computed by ACTUALLY folding every masked candidate at every
               level (context-dependent; the empty-context mean-shift dirs cannot substitute). Leakage-
               free: no held labels, no true answer, no Sigma. The answerability WEIGHT is the binary
               ANSV mask itself (unanswerable candidates are excluded, value 1 inside the pool).
               HARD RULE #1: ALL masked candidates x 4 levels folded (no shortlist); encoder calls
               batched SPINE_CHUNK seqs at a time (non-lossy chunking).
      sigma  = variance-greedy argmax d_c' Sigma d_c -- run ALONGSIDE, reported, NOT required to win.
      random = baseline.
    Per ordering: A = raw mu fold vs B_S* = Sigma-gated fold; Sigma_prev (BEFORE the answer) gates."""
    Wd = decoder.weight.detach(); bd = decoder.bias.detach()
    head = base["headmask"]
    ANSV = (KV[:, :NC] >= 1) & (VV[:, :NC] >= 0)
    arms = ["A"] + [f"B_S{si}" for si in range(len(Sgates))]
    orderings = ("random", "sigma", "spine")
    curves = {o: {a: {"full": [[] for _ in range(q_max + 1)], "tail": [[] for _ in range(q_max + 1)]}
                  for a in arms} for o in orderings}

    def score_step(order_name, arm, q, sc, ch):
        for r, u in enumerate(ch):
            nf = ndcg10(sc[r], list(u["hl"]), u["profset"], head, False)
            nt = ndcg10(sc[r], list(u["hl"]), u["profset"], head, True)
            if nf is not None: curves[order_name][arm]["full"][q].append(nf)
            if nt is not None: curves[order_name][arm]["tail"][q].append(nt)

    t0 = time.time()
    for order_name in orderings:
        rng = np.random.default_rng(rng_seed)
        for b in range(0, len(recs), 256):
            ch = recs[b:b + 256]; B = len(ch)
            pre = None
            if order_name == "random":
                pre = []
                for u in ch:
                    pool = np.flatnonzero(ANSV[u["row"]])
                    perm = rng.permutation(pool)[:q_max].astype(np.int64)
                    if len(perm) < q_max:
                        perm = np.concatenate([perm, -np.ones(q_max - len(perm), np.int64)])
                    pre.append(perm)
            toks = [[] for _ in range(B)]
            asked = np.zeros((B, NC), bool)
            empty = [(np.zeros(0, np.int64), np.zeros(0, np.int64))] * B
            mu, ops = analytic_state(enc, model, empty, dirs_item, dirs_conc, ni)
            muB = {a: mu.clone() for a in arms if a != "A"}
            sc = (mu @ Wd.T + bd).numpy().astype(np.float64)
            for a in arms:
                score_step(order_name, a, 0, sc, ch)
            for q in range(1, q_max + 1):
                if order_name == "random":
                    picks = np.array([pre[r][q - 1] for r in range(B)])
                else:
                    mask = np.zeros((B, NC), bool)                        # answerability gate = the weight
                    for r, u in enumerate(ch):
                        mask[r] = ANSV[u["row"]] & ~asked[r]
                    if order_name == "sigma":
                        qv = ops.quad(dirs_conc).numpy()                  # (B, NC) posterior variance value
                    else:                                                 # SPINE: direct informativeness
                        cand = []; meta = []
                        for r in range(B):
                            base_t = [t[0] for t in toks[r]]; base_l = [t[1] for t in toks[r]]
                            for c in np.flatnonzero(mask[r]):
                                for l in range(4):                        # every non-refuse answer level
                                    cand.append((np.array(base_t + [ni + c], np.int64),
                                                 np.array(base_l + [COFF + l], np.int64)))
                                    meta.append((r, c))
                        qv = np.zeros((B, NC), np.float64)
                        for c0 in range(0, len(cand), SPINE_CHUNK):       # non-lossy chunking
                            sub = cand[c0:c0 + SPINE_CHUNK]
                            # concept-only seqs -> sv=0 IS the convention ('value lives in the LEVEL')
                            mus = fold_mu(enc, sub, [np.zeros(len(s[0]), np.float32) for s in sub], ni)
                            rows = torch.tensor([meta[c0 + k][0] for k in range(len(sub))])
                            sh = (mus - mu[rows]).norm(dim=-1).numpy()    # ||mu(S+(c,l)) - mu(S)||
                            for k in range(len(sub)):
                                r_, c_ = meta[c0 + k]
                                qv[r_, c_] += sh[k] / 4.0                 # mean over the 4 levels
                    qv[~mask] = -np.inf
                    picks = qv.argmax(1)
                    picks[~mask.any(1)] = -1                              # pool exhausted -> skip turn
                for r, u in enumerate(ch):
                    c = int(picks[r])
                    if c < 0:
                        continue
                    asked[r, c] = True
                    lv, _ = concept_answers(u["row"], KV, VV, np.array([c]))
                    if lv[0] != REFUSE:
                        toks[r].append((ni + c, int(lv[0])))
                # concept-only evidence: sv=0 IS the make_pb_example convention ('value lives in the
                # LEVEL') -- analytic_state hard-asserts no item token slips through this zero-sv path
                seqs = [(np.array([t[0] for t in tk], np.int64), np.array([t[1] for t in tk], np.int64))
                        for tk in toks]
                ops_prev = ops                                            # Sigma BEFORE this answer (the gain)
                mu, ops = analytic_state(enc, model, seqs, dirs_item, dirs_conc, ni)
                sc = (mu @ Wd.T + bd).numpy().astype(np.float64)
                score_step(order_name, "A", q, sc, ch)
                for si, Sg in enumerate(Sgates):
                    a = f"B_S{si}"
                    muB[a] = ops_prev.gate(muB[a], mu, Sg)
                    scb = (muB[a] @ Wd.T + bd).numpy().astype(np.float64)
                    score_step(order_name, a, q, scb, ch)
        log(f"[G2] {order_name} ordering done ({(time.time() - t0) / 60:.1f}m)")

    res = {"Sgates": [float(s) for s in Sgates]}
    for o in curves:
        for a in curves[o]:
            f = [round(float(np.mean(x)), 4) if x else float("nan") for x in curves[o][a]["full"]]
            t = [round(float(np.mean(x)), 4) if x else float("nan") for x in curves[o][a]["tail"]]
            res[f"{o}/{a}/full"] = f; res[f"{o}/{a}/tail"] = t
            log(f"[G2] {o:6s} {a:5s} FULL {f}")
            log(f"[G2] {o:6s} {a:5s} TAIL {t}")

    def monotone(xs):
        return all(xs[i + 1] >= xs[i] - 1e-9 for i in range(len(xs) - 1))
    # decisive arm = SPINE (direct informativeness); sigma-greedy REPORTED but never required to win
    a_mono = monotone(res["spine/A/full"]) and monotone(res["spine/A/tail"])
    b_mono = {si: monotone(res[f"spine/B_S{si}/full"]) and monotone(res[f"spine/B_S{si}/tail"])
              for si in range(len(Sgates))}
    best_si = max(range(len(Sgates)), key=lambda si: res[f"spine/B_S{si}/tail"][-1])
    gr_spine = res[f"spine/B_S{best_si}/tail"][4] > res[f"random/B_S{best_si}/tail"][4]
    gr_sigma = res[f"sigma/B_S{best_si}/tail"][4] > res[f"random/B_S{best_si}/tail"][4]  # informational
    above_cold = all(res[f"spine/B_S{best_si}/full"][q] >= res[f"spine/B_S{best_si}/full"][0] - 1e-9
                     for q in range(q_max + 1))                           # full never below cold intercept
    res["g2_A_monotone"] = bool(a_mono)
    res["g2_B_monotone"] = {str(si): bool(v) for si, v in b_mono.items()}
    res["g2_best_S"] = float(Sgates[best_si])
    res["g2_spine_gt_random_q4"] = bool(gr_spine)
    res["g2_sigma_gt_random_q4"] = bool(gr_sigma)                         # reported, NOT a pass condition
    res["g2_full_above_cold"] = bool(above_cold)
    # PASS (reviewer blocking #3): gated monotone + spine>random@q4 + full>=cold. a_mono is REPORTED
    # only -- 'gated monotone where raw isn't' is a CONDITIONAL attribution of credit to the Sigma
    # gate, not a requirement that raw fail; a raw arm that is already monotone is a BETTER outcome
    # (g2_gate_credited records whether the gate was the reason the gated arm is monotone).
    res["g2_gate_credited"] = bool(b_mono[best_si] and not a_mono)
    res["g2_pass"] = bool(b_mono[best_si] and gr_spine and above_cold)
    log(f"[G2] SPINE: A mono={a_mono} (reported, not a pass condition) | B mono={b_mono} | "
        f"best S={Sgates[best_si]:.4g} | gate_credited={res['g2_gate_credited']} | "
        f"spine>random@q4(tail)={gr_spine} | sigma>random@q4(tail)={gr_sigma} (informational) | "
        f"full>=cold={above_cold} => {'PASS' if res['g2_pass'] else 'FAIL'}")
    return res


# ============================= main =============================
def main(args):
    base, ni, enc, decoder = load_frozen(args.base)
    S.PB_NI[0] = ni
    Wd = decoder.weight.detach(); bd = decoder.bias.detach()
    SNAP = trunk_snapshot(enc, decoder)

    # ---- cohorts (HARD RULE #1: full cohorts, no reduction anywhere) ----
    umap, KT, VT = load_answerer("train")
    users = build_pb_users(base, umap)                                    # ALL usable answerer-train users
    SPLv = build_splits(base, SEEDS[0]); vusers = cohort(base, SPLv, "val")
    recs, KV, VV = build_val_recs(base)
    log(f"[data] train users {len(users)} | G0 val cohort {len(vusers)} | gate cohort {len(recs)}")

    # ---- prior variance (the saved bpool2 anisotropic empirical z-variance; file IS the variance) ----
    sig_path = os.path.join(OUT, f"vhead_sig0_{args.base}.npy")
    var_emp = np.load(sig_path) if os.path.exists(sig_path) else empirical_sigma0(enc, users, sig_path)
    log(f"[prior] empirical z-variance: mean {var_emp.mean():.4g} min {var_emp.min():.4g} "
        f"max {var_emp.max():.4g} (Lambda0 = 1/v0 after s0/vfloor fit)")

    # ---- directions + model ----
    dirs_item, dirs_conc, dirs_conc_mode = build_dirs(enc, decoder, ni, args.members)
    if args.conc_dirs:                                        # override concept dirs with precomputed encoder MEAN-SHIFT dirs
        md = np.load(args.conc_dirs).astype(np.float64)       # (NC,512) unit rows = normalize(mu({c})-mu(empty))
        assert md.shape[0] == dirs_conc.shape[0] and md.shape[1] == D, f"conc_dirs shape {md.shape} != {tuple(dirs_conc.shape)}"
        nrm = np.linalg.norm(md, axis=1); assert np.allclose(nrm[nrm>0], 1.0, atol=1e-4), "conc_dirs must be unit-norm"
        dirs_conc = torch.tensor(md, dtype=dirs_conc.dtype); dirs_conc_mode = "encoder_mean_shift"
        log(f"[dirs] OVERRIDE concept dirs = encoder mean-shift from {args.conc_dirs} (leverage-corrected)")
    model = PrecAcc(var_emp)
    log(f"[pa] fitted scalars: {sum(p.numel() for p in model.parameters())} (alpha 2x{N_BUCK} + s0 + vfloor + tau2)")

    # ---- G0 anchor BEFORE anything runs (external canary; mu path untouched but assert anyway) ----
    f0, t0 = eval_student(enc, None, base, SPLv, vusers, Wd, bd)
    log(f"[G0] frozen-mu eval_student full={f0:.6f} tail={t0:.6f} (expected ~{EXPECT_FULL}/{EXPECT_TAIL})")
    assert abs(f0 - EXPECT_FULL) < 1e-4 and abs(t0 - EXPECT_TAIL) < 1e-4, \
        f"G0 ANCHOR FAIL: {f0:.6f}/{t0:.6f} != {EXPECT_FULL}/{EXPECT_TAIL} (trunk mis-load / wrong protocol)"

    # ---- alpha-fit: Delta_e precompute -> MANDATORY sign proof -> increment fit (or resume) ----
    ckp = os.path.join(OUT, args.tag + ".pt")
    hist = []; sign = None
    if args.resume and os.path.exists(ckp):
        blob = torch.load(ckp, map_location="cpu")
        model.load_state_dict(blob["precacc"]); hist = blob.get("fit_hist", [])
        sign = blob.get("sign_proof")
        log(f"[pa] RESUMED fitted scalars from {ckp}")
    else:
        fit_users = users
        if args.max_users and args.max_users < len(users):   # HARD RULE #1 sanctioned subsample (fit only)
            ru = np.random.default_rng(SEEDS[0])
            fit_users = [users[i] for i in sorted(ru.choice(len(users), args.max_users, replace=False).tolist())]
            log(f"[data] *** HARD RULE #1 SANCTIONED SUBSAMPLE (author sign-off): alpha-fit on "
                f"{len(fit_users)}/{len(users)} users; max_atoms={args.max_atoms or 'all'}. Gates/G0 use FULL cohorts. ***")
        sub_tag = "" if (not args.max_users and not args.max_atoms) else f"_u{args.max_users}_a{args.max_atoms}"
        inc_path = args.inc_cache or os.path.join(OUT, f"inc_{args.base}{sub_tag}.pt")
        inc = precompute_increments(enc, decoder, fit_users, KT, VT, ni, inc_path, max_atoms=args.max_atoms)
        sign = prefit_sign_proof(model, decoder, inc, dirs_item, dirs_conc, ni, args.huber)  # gate: G > 0
        hist = fit_alphas(model, decoder, inc, dirs_item, dirs_conc, ni, args)
    model.eval()

    # ---- freeze + G0 re-check after the fit (the fit CANNOT touch the trunk; prove it) ----
    dr = trunk_drift(enc, decoder, SNAP)
    f1, t1 = eval_student(enc, None, base, SPLv, vusers, Wd, bd)
    log(f"[G0 post-fit] drift={dr:.2e} mu-eval full={f1:.6f} tail={t1:.6f} "
        f"bit-identical={'PASS' if (dr == 0.0 and f1 == f0 and t1 == t0) else 'FAIL'}")
    assert dr == 0.0 and f1 == f0 and t1 == t0, "G0 FAIL post-fit: frozen mean moved"

    # ---- gates ----
    v0 = model.v0().detach().numpy()
    Sgates = [a_ * float(np.mean(v0)) for a_ in (0.25, 1.0, 4.0)] if args.gate_s <= 0 else [args.gate_s]
    g1 = gate_g1(enc, model, decoder, recs, ni, dirs_item, dirs_conc)
    g2 = gate_g2(enc, model, decoder, base, recs, KV, VV, ni, dirs_item, dirs_conc, Sgates)

    blob = {"precacc": model.state_dict(), "base": args.base, "fit_hist": hist,
            "objective": "increment_huber", "huber_beta": args.huber, "sign_proof": sign,
            "alphas": model.alphas().detach().numpy(),
            "v0_mean": float(np.mean(v0)), "tau2": float(model.tau2()),
            "full": f1, "tail": t1, "g1": g1, "g2": g2,
            "members": args.members or "NONE_fallback",
            "dirs_conc_mode": dirs_conc_mode}
    safe_save(blob, ckp)
    safe_save(blob, os.path.join(OUT, f"{args.tag}_final.pt"))            # save-all rule
    log(f"[pa] done. G1c rho={g1['g1c_rho']} ({'PASS' if g1['g1c_pass'] else 'FAIL'}) | "
        f"G2 {'PASS' if g2.get('g2_pass') else 'FAIL'} | saved {ckp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="pbC_best")     # frozen trunk checkpoint stem under .cache/set_mn/
    ap.add_argument("--tag", default="precacc")
    ap.add_argument("--epochs", type=int, default=2)  # full passes of the 13-scalar fit (all train users)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--huber", type=float, default=1.0)   # smooth-L1 beta on the Delta_v - Delta_e residual
    ap.add_argument("--inc_cache", default="")        # Delta_e cache path; default .cache/set_mn/inc_<base>.pt
    ap.add_argument("--members", default="")          # npz concept->item members -> EXACT whitened decoder-space
    ap.add_argument("--conc_dirs", default="")        # .npy (NC,512) unit encoder mean-shift dirs -> OVERRIDES whitened (leverage fix)
                                                      # d_c; omit -> raw normalized Ec fallback (mode recorded)
    ap.add_argument("--gate_s", type=float, default=0.0)   # 0 => grid {0.25,1,4} x mean(v0)
    ap.add_argument("--resume", action="store_true")
    # HARD RULE #1 SANCTIONED reductions for the alpha-fit ONLY (the fit is a <=13-scalar regression; a
    # user/atom subsample gives statistically identical alpha's and bounds the O(sum m(m-1)) Delta_e cost
    # + the growing-out-list memory). 0 = exact (all users / all atoms). Gates + G0 always use full cohorts.
    ap.add_argument("--max_users", type=int, default=0)     # subsample train users for the Delta_e fit (0=all)
    ap.add_argument("--max_atoms", type=int, default=0)     # cap LOO-evaluated atoms per user (0=all; folds still use full S)
    main(ap.parse_args())
