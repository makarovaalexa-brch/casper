"""train_varhead.py -- STAGE-1 "VarHead": low-rank Gaussian posterior HEAD on the FROZEN pbC set-encoder.
Design sheet: casper/DESIGN_SHEET_VARHEAD.md (2026-07-19). Partial-Mult-VAE: the MEAN is literally the
frozen pbC point embedding (mu = z0 + head(r_pool), untouched); we train ONLY {W_V, W_d, logSig0} so that
Sigma = V V^T + diag(softplus(logd)), V = W_V(r_pool).reshape(512, 32), logd = W_d(r_pool),
read from the SAME pooled vector r_pool = cat[PMA pool, log1p(|set|)] the frozen mean head reads.

OBJECTIVE (partial ELBO, likelihood = the REAL multinomial decoder):
    E_eps[ NLL_mult(held-liked | z' = mu + V eps1 + sqrt(diag) eps2) ]
        + beta * KL( N(mu, Sigma) || N(z0, Sigma0) ),   beta annealed 0 -> beta_max.
KL via the matrix-determinant lemma / Woodbury (32x32 solves only, never a dense 512x512 inverse).

GATES (computed + logged every epoch):
  G0  eval_student on mu must be BIT-IDENTICAL to the frozen pbC numbers (expected 0.4946 / 0.3372)
      every epoch, and max|trunk+decoder param drift| == 0 (hard assert, train_concepts_ord pattern).
  G1  (a) tr(Sigma) strictly decreasing in evidence count k in {0,1,2,4,8,16,32,full};
      (b) DIRECTIONAL: folding concept c cuts RELATIVE variance ((q0-q1)/q0) along its own direction d_c
          >= 2x more than along OTHER concepts' directions (relative + data-direction null, so a pure
          set-size multiplicative shrink cannot pass);
      (c) calibration: Spearman rho( sqrt(w^T Sigma w), realized held-item NLL ) >= 0.2.
  G2  (decisive) cold-start per-step q=0..8, truthful answers from the val answerer tables, REAL decoder,
      FULL + TAIL: (A) raw mu fold vs (B) Sigma-gated fold mu' = mu' + Sigma(Sigma+S)^-1 (mu_new - mu');
      question order variance-greedy (argmax d_c^T Sigma d_c) vs random, BOTH drawn from the SAME
      per-user answerability-masked pool (else the pass could come from the E0 answerability channel).
      PASS = B monotone-nonneg where A is not, AND greedy > random by q4.

HARD RULES: NO data reduction of any kind (all usable answerer-train users; full val cohort in every gate);
FULL + TAIL always, scored with z @ decoder.weight.T + decoder.BIAS (the LEARNED bias -- NEVER popb);
CPU only; no LLM calls. 300-study users quarantined (never in these cohorts).
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
from set_mn import SetEncoder, RSD, eval_student, make_batches, pad_batch, build_pb_users, \
    make_pb_example, load_answerer, concept_answers, sv_to_level
from signed_latent import load_arena_base, build_splits, cohort, ndcg10, safe_save, scale_rating, SEEDS, LO, log
import arena_core as AC

S.set_grading("halfstar")                      # pbC_best.pt is HALFSTAR grading (NLEV=15, gamma[15]); ordinal(NLEV=5) mis-sizes the FiLM
NC = S.NC; NLEV = S.NLEV; REFUSE = S.LV_REFUSE
META = "C:/dev/phd/casper/data/movielens/.cache/ml25m/meta.npz"
OUT = "C:/dev/phd/casper/.cache/set_mn"; os.makedirs(OUT, exist_ok=True)
D = 512; RANK = 32; DIN = D + 1                # r_pool = cat[p (512), log1p(|set|) (1)]
EXPECT_FULL, EXPECT_TAIL = 0.4946, 0.3372      # frozen pbC mean-head canary values (G0)
DFLOOR = 1e-4                                  # variance floor: Dm = softplus(logd) + DFLOOR at EVERY site
                                               # (design-sheet requirement; without it the beta~0 anneal phase
                                               # drives Dm->0 and kl_lowrank's V/Dm, log(Dm) -> 0*inf = NaN)


# ============================= VarHead (the ONLY trainable module) =============================
class VarHead(nn.Module):
    """Sigma = V V^T + diag(softplus(logd)); V = W_V(r_pool).view(512,32); logd = W_d(r_pool).
    Init: W_V == 0 (weight AND bias) and W_d weight == 0 with bias = softplus^-1(Sigma0 diag), so at init
    Sigma(any input) == Sigma0 exactly (empty set included). Sigma0 = learned diagonal, init from the
    empirical variance of pbC full-profile beliefs over train users."""
    def __init__(self, d=D, r=RANK, din=DIN):
        super().__init__()
        self.d = d; self.r = r
        self.W_V = nn.Linear(din, d * r)
        self.W_d = nn.Linear(din, d)
        nn.init.zeros_(self.W_V.weight); nn.init.zeros_(self.W_V.bias)
        nn.init.zeros_(self.W_d.weight); nn.init.zeros_(self.W_d.bias)   # bias overwritten by set_sigma0
        self.logSig0 = nn.Parameter(torch.zeros(d))                       # prior diag = softplus(logSig0)

    @staticmethod
    def softplus_inv(x):
        # inverse of softplus for x > 0: log(exp(x) - 1) = x + log1p(-exp(-x))
        return x + torch.log(-torch.expm1(-x))

    def set_sigma0(self, var_diag):
        """var_diag (d,) empirical variance of full-profile z. Sets BOTH the learned prior Sigma0 and the
        W_d bias so Sigma(empty) ~ Sigma0 at init."""
        # subtract the floor before inverting so softplus(.)+DFLOOR reproduces var_diag exactly at init
        v = torch.clamp(torch.as_tensor(var_diag, dtype=torch.float32) - DFLOOR, min=1e-6)
        with torch.no_grad():
            self.logSig0.copy_(self.softplus_inv(v))
            self.W_d.bias.copy_(self.softplus_inv(v))

    def forward(self, rpool):
        B = rpool.shape[0]
        V = self.W_V(rpool).view(B, self.d, self.r)      # (B, 512, 32)
        logd = self.W_d(rpool)                           # (B, 512)
        return V, logd

    def sigma0(self):
        return F.softplus(self.logSig0) + DFLOOR         # (512,) >= DFLOOR (same floor as Dm)


def kl_lowrank(V, logd, mu, z0, vh):
    """KL( N(mu, VV^T + Dm) || N(z0, diag(s0)) ), Dm = softplus(logd) + DFLOOR (floored -- NLL at beta~0
    would otherwise drive Dm->0 and V/Dm, log(Dm) become 0*inf = NaN).
    logdet Sigma via the matrix-determinant lemma: logdet(Dm) + logdet(I_r + V^T Dm^-1 V) -- 32x32 only."""
    Dm = F.softplus(logd) + DFLOOR                                    # (B, d) >= DFLOOR
    s0 = vh.sigma0()                                                  # (d,)
    tr = ((Dm + (V ** 2).sum(-1)) / s0).sum(-1)                       # tr(Sigma0^-1 Sigma)
    diff = mu - z0                                                    # mu, z0 both frozen -> trains s0 only
    mah = ((diff ** 2) / s0).sum(-1)
    logdet0 = torch.log(s0).sum()
    Vd = V / Dm.unsqueeze(-1)                                         # Dm^-1 V
    M = torch.eye(V.shape[-1]).unsqueeze(0) + torch.einsum("bdr,bds->brs", Vd, V)   # (B, r, r)
    logdetS = torch.log(Dm).sum(-1) + torch.linalg.slogdet(M)[1]
    return 0.5 * (tr + mah - V.shape[1] + logdet0 - logdetS)          # (B,)


def quad_form(V, Dm, Wdir):
    """w^T Sigma w for a batch of users x a bank of directions. V (B,d,r), Dm (B,d), Wdir (N,d) -> (B,N)."""
    q_diag = Dm @ (Wdir ** 2).T                                       # (B, N)
    VtW = torch.einsum("bdr,nd->bnr", V, Wdir)                        # (B, N, r)
    return q_diag + (VtW ** 2).sum(-1)


def gated_step(mu_prev, mu_new, V, Dm, Sgate):
    """Sigma-gated fold: mu' = mu_prev + Sigma (Sigma + S I)^-1 (mu_new - mu_prev), Woodbury (32x32 solve)."""
    x = mu_new - mu_prev                                              # (B, d)
    Dt = Dm + Sgate                                                   # (B, d) = diag(Sigma + S I) part
    xd = x / Dt
    Vd = V / Dt.unsqueeze(-1)                                         # (Dm + S)^-1 V
    M = torch.eye(V.shape[-1]).unsqueeze(0) + torch.einsum("bdr,bds->brs", Vd, V)
    sol = torch.linalg.solve(M, torch.einsum("bdr,bd->br", V, xd).unsqueeze(-1)).squeeze(-1)
    y = xd - torch.einsum("bdr,br->bd", Vd, sol)                      # (Sigma + S I)^-1 x
    Sy = Dm * y + torch.einsum("bdr,br->bd", V, torch.einsum("bdr,bd->br", V, y))    # Sigma y
    return mu_prev + Sy


def spearman(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    ra = np.argsort(np.argsort(a)).astype(np.float64); rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    den = math.sqrt(float((ra * ra).sum()) * float((rb * rb).sum())) + 1e-12
    return float((ra * rb).sum() / den)


# ============================= frozen trunk + r_pool capture =============================
_POOL = {}
def _pool_hook(mod, inp, out):
    _POOL["r"] = inp[0].detach()                                      # EXACT input the frozen mean head sees


def load_frozen(pbc_stem):
    """pbC SetEncoder + decoder, everything frozen; hook on enc.head captures r_pool bit-exactly."""
    base = load_arena_base(); ni = base["ni"]
    ck = torch.load(os.path.join(OUT, pbc_stem + ".pt"), map_location="cpu")
    NT = ck["student"]["item_emb.weight"].shape[0]
    assert NT == ni + NC, f"token count {NT} != ni+NC {ni + NC}"
    enc = SetEncoder(NT, token_mode="film", pool="attn", nlev=NLEV, nknow=0)   # concept_eval protocol
    missing, unexpected = enc.load_state_dict(ck["student"], strict=False)
    # G0 must not be self-referential: a silently mis-loaded trunk (wrong token_mode/stem) would give a
    # wrong-but-stable f0 that "passes" bit-identity forever. Hard-fail on ANY missing key; unexpected keys
    # (extra checkpoint entries the film/attn config doesn't own) are listed so a mismatch is visible.
    assert len(missing) == 0, f"[load] trunk mis-load: MISSING keys {missing}"
    if unexpected:
        log(f"[load] unexpected checkpoint keys (not loaded): {unexpected}")
    decoder = nn.Linear(D, ni); decoder.load_state_dict(ck["decoder"])
    for p_ in enc.parameters():
        p_.requires_grad_(False)
    for p_ in decoder.parameters():
        p_.requires_grad_(False)
    enc.eval(); decoder.eval()
    enc.head.register_forward_hook(_pool_hook)
    log(f"[load] {pbc_stem}: tokens={NT} full={ck.get('full')} tail={ck.get('tail')} "
        f"missing={len(missing)} unexpected={len(unexpected)} | trunk+decoder FROZEN")
    return base, ni, enc, decoder


def encode_pool(enc, ids, vals, pad, lvs):
    """One frozen forward -> (mu, r_pool). mu IS the frozen point embedding; nothing is re-derived."""
    with torch.no_grad():
        mu = enc(ids, vals, pad, lvs)
    return mu, _POOL["r"]


def trunk_snapshot(enc, decoder):
    snap = {("enc." + k): v.detach().clone() for k, v in enc.state_dict().items()}
    snap.update({("dec." + k): v.detach().clone() for k, v in decoder.state_dict().items()})
    return snap


def trunk_drift(enc, decoder, snap):
    m = 0.0
    for k, v in enc.state_dict().items():
        m = max(m, float((v - snap["enc." + k]).abs().max()))
    for k, v in decoder.state_dict().items():
        m = max(m, float((v - snap["dec." + k]).abs().max()))
    return m


# ============================= Sigma0 init (ALL train users, cached) =============================
def empirical_sigma0(enc, users, cache_path):
    """Empirical per-dim variance of pbC FULL-PROFILE beliefs over ALL train users (no reduction).
    Streaming sum / sumsq -- never materialises the (150k, 512) matrix. Cached (one-time cost)."""
    if os.path.exists(cache_path):
        v = np.load(cache_path)
        log(f"[sig0] loaded cached empirical variance {cache_path} (mean {v.mean():.4g})")
        return v
    log(f"[sig0] computing empirical full-profile z variance over {len(users)} train users ...")
    order = np.argsort(np.array([len(u["items"]) for u in users]))
    batches = make_batches(users, order)
    s = torch.zeros(D, dtype=torch.float64); s2 = torch.zeros(D, dtype=torch.float64); n = 0
    t0 = time.time()
    with torch.no_grad():
        for bi, bat in enumerate(batches):
            ids, vals, pad = pad_batch(users, bat)
            lvs = torch.from_numpy(sv_to_level(vals.numpy()))         # padded rows masked out by pad
            z = enc(ids, vals, pad, lvs).double()
            s += z.sum(0); s2 += (z ** 2).sum(0); n += z.shape[0]
            if (bi + 1) % 50 == 0:
                log(f"  [sig0] batch {bi + 1}/{len(batches)} ({(time.time() - t0) / 60:.1f}m)")
    var = (s2 / n - (s / n) ** 2).clamp_min(1e-6).float().numpy()
    np.save(cache_path, var)
    log(f"[sig0] done: mean {var.mean():.4g} min {var.min():.4g} max {var.max():.4g} -> {cache_path}")
    return var


# ============================= val cohort (concept_eval protocol, NOT the nan cmd_pb pairing) =============================
def build_val_recs(base):
    """Val answerer cohort with the AC.SEED-keyed per-user half split (identical to concept_eval.py):
    evidence = known half (items + ratings, for G1), profset = known half, targets = held-liked (>=LO).
    Full cohort, no subsampling."""
    head = base["headmask"]
    d = np.load(META); uu = d["uu"].astype(np.int64); ii = d["ii"].astype(np.int64); rr = d["rr"].astype(np.float64)
    o = np.argsort(uu, kind="stable"); uu, ii, rr = uu[o], ii[o], rr[o]
    bnd = np.searchsorted(uu, np.arange(uu[-1] + 2))
    uidsV = np.load(RSD + "/mm_val_uids.npy")
    KV = np.load(RSD + "/mm_val_know.npy"); VV = np.load(RSD + "/mm_val_val.npy")
    recs = []
    for r, uid in enumerate(uidsV):
        a, b = bnd[int(uid)], bnd[int(uid) + 1]
        its, rat = ii[a:b], rr[a:b]
        if len(its) < 8:
            continue
        ru = np.random.default_rng(AC.SEED * 1_000_003 + int(uid))
        p = ru.permutation(len(its)); h = len(its) // 2
        ki, kr = its[p[:h]], rat[p[:h]]
        hi, hr = its[p[h:]], rat[p[h:]]
        hl = hi[hr >= LO]
        if len(ki) < 4 or len(hl) == 0 or (~head[hl]).sum() == 0:
            continue
        recs.append(dict(row=r, ki=ki.astype(np.int64), ksv=scale_rating(kr).astype(np.float32),
                         profset=set(int(x) for x in ki), hl=hl.astype(np.int64)))
    log(f"[val] usable eval users {len(recs)} (concept_eval split, full cohort)")
    return recs, KV, VV


def batch_forward_sigma(enc, vh, seqs):
    """seqs = list of (ids_np, lv_np) token sequences (possibly length 0). One padded frozen forward
    -> mu (B,512), V (B,512,32), Dm (B,512). Empty sequences follow the concept_eval intercept convention
    (a single all-padded token)."""
    B = len(seqs); L = max(1, max(len(s[0]) for s in seqs))
    ids = np.zeros((B, L), np.int64); lv = np.zeros((B, L), np.int64); pad = np.ones((B, L), bool)
    for r, (tid, tlv) in enumerate(seqs):
        n = len(tid)
        if n:
            ids[r, :n] = tid; lv[r, :n] = tlv; pad[r, :n] = False
    mu, rpool = encode_pool(enc, torch.from_numpy(ids), torch.zeros((B, L)),
                            torch.from_numpy(pad), torch.from_numpy(lv))
    with torch.no_grad():
        V, logd = vh(rpool)
        Dm = F.softplus(logd) + DFLOOR
    return mu, V, Dm


# ============================= G1: posterior usable =============================
def gate_g1(enc, vh, decoder, base, recs, KV, VV, ni, rng_seed=7):
    Wd = decoder.weight.detach(); bd = decoder.bias.detach()
    rng = np.random.default_rng(rng_seed)
    res = {}
    # ---------- (a) tr(Sigma) vs evidence count k (item evidence from the known half) ----------
    ks = [0, 1, 2, 4, 8, 16, 32, "full"]
    traces = {}
    for k in ks:
        acc = []; t0 = time.time()
        for b in range(0, len(recs), 256):
            ch = recs[b:b + 256]
            seqs = []
            for u in ch:
                if k == 0:
                    seqs.append((np.zeros(0, np.int64), np.zeros(0, np.int64)))
                    continue
                n = len(u["ki"]) if k == "full" else min(int(k), len(u["ki"]))
                sel = np.arange(len(u["ki"])) if k == "full" else rng.choice(len(u["ki"]), size=n, replace=False)
                seqs.append((u["ki"][sel], sv_to_level(u["ksv"][sel])))
            _, V, Dm = batch_forward_sigma(enc, vh, seqs)
            acc.append((Dm.sum(-1) + (V ** 2).sum((-1, -2))).numpy())
        traces[str(k)] = float(np.mean(np.concatenate(acc)))
    vals_ = [traces[str(k)] for k in ks]
    g1a = all(vals_[i] > vals_[i + 1] for i in range(len(vals_) - 1))
    res["g1a_traces"] = {str(k): round(traces[str(k)], 4) for k in ks}
    res["g1a_pass"] = bool(g1a)
    log(f"[G1a] tr(Sigma) by k: {res['g1a_traces']}  strictly-decreasing={'PASS' if g1a else 'FAIL'}")

    # ---------- (b) DIRECTIONAL: fold concept c -> RELATIVE variance drop along d_c vs OTHER concepts' dirs ----
    # RELATIVE drops ((q0-q1)/q0) with the null = other concepts' directions d_{c'!=c}. Absolute drops vs random
    # Gaussian probes are fooled by a pure set-size Sigma: multiplicative shrink gives drop ~ prior variance
    # along the direction, and concept (data) directions carry more prior variance than isotropic probes under
    # the anisotropic empirical Sigma0 -- so >=2x could pass with ZERO directionality. Relative drops are
    # invariant to that (multiplicative shrink -> identical relative drop everywhere -> ratio 1 -> FAIL).
    with torch.no_grad():
        Ec = enc.item_emb.weight[ni:ni + NC].detach()
        d_c = Ec / Ec.norm(dim=-1, keepdim=True).clamp_min(1e-8)                    # (NC, 512)
        _, V0, D0 = batch_forward_sigma(enc, vh, [(np.zeros(0, np.int64), np.zeros(0, np.int64))])
        rel_own = np.zeros(NC); rel_oth = np.zeros(NC)
        q0_all = quad_form(V0, D0, d_c)[0].clamp_min(1e-12)                         # (NC,) prior var along d_c
        for b in range(0, NC, 256):
            cids = np.arange(b, min(b + 256, NC))
            seqs = [(np.array([ni + c], np.int64), np.array([S.CLEVEL_OFFSET + 3], np.int64)) for c in cids]
            _, V1, D1 = batch_forward_sigma(enc, vh, seqs)                          # fold c at "loved"
            q1 = quad_form(V1, D1, d_c)                                             # (B, NC) posterior var along ALL d_c'
            R = ((q0_all.unsqueeze(0) - q1) / q0_all.unsqueeze(0)).numpy()          # relative drop matrix
            diag = R[np.arange(len(cids)), cids]
            rel_own[cids] = diag
            rel_oth[cids] = (R.sum(1) - diag) / max(NC - 1, 1)                      # null: mean over c' != c
    mo, mr = float(rel_own.mean()), float(rel_oth.mean())
    ratio = mo / max(mr, 1e-12) if mr > 0 else (float("inf") if mo > 0 else 0.0)
    res["g1b_rel_own"] = round(mo, 6); res["g1b_rel_other"] = round(mr, 6)
    res["g1b_ratio"] = round(ratio, 3); res["g1b_pass"] = bool(mo > 0 and ratio >= 2.0)
    log(f"[G1b] directional (ALL {NC} concepts, RELATIVE drops): own-dir {mo:.3g} vs other-concept-dir {mr:.3g} "
        f"ratio {ratio:.2f} ({'PASS >=2x' if res['g1b_pass'] else 'FAIL'})")

    # ---------- (c) calibration: sqrt(w^T Sigma w) vs realized held-item NLL at mu ----------
    unc = []; err = []
    with torch.no_grad():
        for b in range(0, len(recs), 256):
            ch = recs[b:b + 256]
            seqs = []
            for u in ch:
                n = min(8, len(u["ki"]))
                sel = rng.choice(len(u["ki"]), size=n, replace=False)
                seqs.append((u["ki"][sel], sv_to_level(u["ksv"][sel])))
            mu, V, Dm = batch_forward_sigma(enc, vh, seqs)
            logsm = F.log_softmax(mu @ Wd.T + bd, -1)
            for r, u in enumerate(ch):
                w = Wd[u["hl"]].mean(0)
                w = w / w.norm().clamp_min(1e-8)
                unc.append(float(torch.sqrt(quad_form(V[r:r + 1], Dm[r:r + 1], w.unsqueeze(0))[0, 0])))
                err.append(float(-logsm[r, u["hl"]].mean()))
    rho = spearman(unc, err)
    res["g1c_rho"] = round(rho, 4); res["g1c_pass"] = bool(rho >= 0.2)
    log(f"[G1c] calibration Spearman rho(sqrt(w'Sw), held NLL) = {rho:.4f} over {len(unc)} users "
        f"({'PASS >=0.2' if res['g1c_pass'] else 'FAIL' if rho >= 0.05 else 'FAIL (<0.05: fail-direction)'})")
    return res


# ============================= G2: decisive cold-start curves =============================
def gate_g2(enc, vh, decoder, base, recs, KV, VV, ni, Sgates, q_max=8, rng_seed=11):
    """Per-step q=0..q_max cold-start, truthful concept answers. BOTH orderings draw from the SAME
    per-user answerability-masked pool (ANSV): otherwise greedy's privileged ANSV mask vs random-over-all-NC
    (refusals burning turns) would let the pass come from the answerability channel alone (the E0 effect),
    proving nothing about Sigma. So greedy > random isolates the ORDERING (Sigma) signal. Users whose
    answerable pool is exhausted SKIP the turn (no token appended, never a duplicate). Arms x orderings:
      A = raw mu fold (frozen pbC point), B = Sigma-gated fold (one B curve per gate scale S).
    ALL val recs users, FULL + TAIL NDCG@10 with the REAL decoder bias every step."""
    Wd = decoder.weight.detach(); bd = decoder.bias.detach()
    head = base["headmask"]
    with torch.no_grad():
        Ec = enc.item_emb.weight[ni:ni + NC].detach()
        d_c = Ec / Ec.norm(dim=-1, keepdim=True).clamp_min(1e-8)                    # question-value directions
    ANSV = (KV[:, :NC] >= 1) & (VV[:, :NC] >= 0)                                    # answerability (val tables)
    arms = ["A"] + [f"B_S{si}" for si in range(len(Sgates))]
    curves = {o: {a: {"full": [[] for _ in range(q_max + 1)], "tail": [[] for _ in range(q_max + 1)]}
                  for a in arms} for o in ("random", "greedy")}

    def score_step(order_name, arm, q, sc, ch):
        for r, u in enumerate(ch):
            nf = ndcg10(sc[r], list(u["hl"]), u["profset"], head, False)
            nt = ndcg10(sc[r], list(u["hl"]), u["profset"], head, True)
            if nf is not None: curves[order_name][arm]["full"][q].append(nf)
            if nt is not None: curves[order_name][arm]["tail"][q].append(nt)

    t0 = time.time()
    for order_name in ("random", "greedy"):
        rng = np.random.default_rng(rng_seed)
        for b in range(0, len(recs), 256):
            ch = recs[b:b + 256]; B = len(ch)
            pre = None
            if order_name == "random":
                # SAME answerability-masked pool as greedy (de-confounds the E0 answerability channel);
                # pad with -1 (= skip turn) when a user has fewer than q_max answerable concepts
                pre = []
                for u in ch:
                    pool = np.flatnonzero(ANSV[u["row"]])
                    perm = rng.permutation(pool)[:q_max].astype(np.int64)
                    if len(perm) < q_max:
                        perm = np.concatenate([perm, -np.ones(q_max - len(perm), np.int64)])
                    pre.append(perm)
            toks = [[] for _ in range(B)]                                           # answered (or refused-padded) tokens
            asked = np.zeros((B, NC), bool)
            # ----- q = 0: intercept (empty set) -----
            mu, V, Dm = batch_forward_sigma(enc, vh, [(np.zeros(0, np.int64), np.zeros(0, np.int64))] * B)
            muB = {a: mu.clone() for a in arms if a != "A"}                         # gated beliefs start at mu(empty)
            sc = (mu @ Wd.T + bd).numpy().astype(np.float64)
            for a in arms:
                score_step(order_name, a, 0, sc, ch)
            for q in range(1, q_max + 1):
                # ----- pick the next question -----
                if order_name == "random":
                    picks = np.array([pre[r][q - 1] for r in range(B)])
                else:
                    qv = quad_form(V, Dm, d_c).numpy()                              # (B, NC) current variance value
                    mask = np.zeros((B, NC), bool)
                    for r, u in enumerate(ch):
                        mask[r] = ANSV[u["row"]] & ~asked[r]                        # answerability-masked greedy
                    qv[~mask] = -np.inf
                    picks = qv.argmax(1)
                    picks[~mask.any(1)] = -1                                        # pool exhausted -> skip turn
                                                                                    # (argmax over all -inf would
                                                                                    # return 0 = a duplicate token)
                # ----- truthful answers; append token (pick -1 = skipped turn; refusal dropped = burnt) -----
                for r, u in enumerate(ch):
                    c = int(picks[r])
                    if c < 0:
                        continue
                    asked[r, c] = True
                    lv, _ = concept_answers(u["row"], KV, VV, np.array([c]))
                    if lv[0] != REFUSE:
                        toks[r].append((ni + c, int(lv[0])))
                # ----- one frozen forward on the accumulated tokens (shared by A and B) -----
                seqs = [(np.array([t[0] for t in tk], np.int64), np.array([t[1] for t in tk], np.int64))
                        for tk in toks]
                V_prev, Dm_prev = V, Dm                                             # Sigma BEFORE this answer (gain)
                mu, V, Dm = batch_forward_sigma(enc, vh, seqs)
                sc = (mu @ Wd.T + bd).numpy().astype(np.float64)
                score_step(order_name, "A", q, sc, ch)                              # A = raw mu fold
                for si, Sg in enumerate(Sgates):
                    a = f"B_S{si}"
                    muB[a] = gated_step(muB[a], mu, V_prev, Dm_prev, Sg)            # B = Sigma-gated fold
                    scb = (muB[a] @ Wd.T + bd).numpy().astype(np.float64)
                    score_step(order_name, a, q, scb, ch)
        log(f"[G2] {order_name} ordering done ({(time.time() - t0) / 60:.1f}m)")

    # ----- aggregate + pass checks -----
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
    a_mono = monotone(res["greedy/A/full"]) and monotone(res["greedy/A/tail"])
    b_mono = {si: monotone(res[f"greedy/B_S{si}/full"]) and monotone(res[f"greedy/B_S{si}/tail"])
              for si in range(len(Sgates))}
    best_si = max(range(len(Sgates)), key=lambda si: res[f"greedy/B_S{si}/tail"][-1])
    gr = res[f"greedy/B_S{best_si}/tail"][4] > res[f"random/B_S{best_si}/tail"][4]
    res["g2_A_monotone"] = bool(a_mono)
    res["g2_B_monotone"] = {str(si): bool(v) for si, v in b_mono.items()}
    res["g2_best_S"] = float(Sgates[best_si])
    res["g2_greedy_gt_random_q4"] = bool(gr)
    res["g2_pass"] = bool(b_mono[best_si] and (not a_mono) and gr)
    log(f"[G2] A monotone={a_mono} | B monotone={b_mono} | best S={Sgates[best_si]:.4g} | "
        f"greedy>random@q4(tail)={gr} => {'PASS' if res['g2_pass'] else 'FAIL'}")
    return res


# ============================= training =============================
def main(args):
    base, ni, enc, decoder = load_frozen(args.base)
    S.PB_NI[0] = ni
    Wd = decoder.weight.detach(); bd = decoder.bias.detach()
    SNAP = trunk_snapshot(enc, decoder)

    # ---- cohorts (HARD RULE #1: full cohorts, no reduction anywhere) ----
    umap, KT, VT = load_answerer("train")
    users = build_pb_users(base, umap)                                # ALL usable answerer-train users
    SPLv = build_splits(base, SEEDS[0]); vusers = cohort(base, SPLv, "val")   # canonical G0 cohort
    recs, KV, VV = build_val_recs(base)                               # concept_eval-aligned gate cohort
    log(f"[data] train users {len(users)} | G0 val cohort {len(vusers)} | gate cohort {len(recs)}")

    # ---- VarHead + Sigma0 init ----
    vh = VarHead()
    sig0 = empirical_sigma0(enc, users, os.path.join(OUT, f"vhead_sig0_{args.base}.npy"))
    vh.set_sigma0(sig0)
    n_par = sum(p.numel() for p in vh.parameters())
    log(f"[vh] trainables W_V/W_d/logSig0 only: {n_par / 1e6:.2f}M params (rank {RANK})")
    opt = torch.optim.AdamW(vh.parameters(), lr=args.lr, weight_decay=0.0)   # SINGLE group, new params only

    start_ep = 0
    ckp = os.path.join(OUT, args.tag + ".pt")
    if args.resume and os.path.exists(ckp):
        blob = torch.load(ckp, map_location="cpu")
        vh.load_state_dict(blob["varhead"]); start_ep = blob.get("epoch", 0)
        if "opt" in blob:
            opt.load_state_dict(blob["opt"])
        log(f"[vh] RESUMED ep{start_ep}")

    # ---- G0 reference: frozen mean must be bit-identical forever ----
    f0, t0 = eval_student(enc, None, base, SPLv, vusers, Wd, bd)
    log(f"[G0] INIT frozen-mu eval_student full={f0:.6f} tail={t0:.6f} (expected ~{EXPECT_FULL}/{EXPECT_TAIL})")
    # anchor f0/t0 to the EXTERNAL canary values BEFORE training -- bit-identity alone is self-referential
    assert abs(f0 - EXPECT_FULL) < 1e-4 and abs(t0 - EXPECT_TAIL) < 1e-4, \
        f"G0 ANCHOR FAIL: frozen-mu eval {f0:.6f}/{t0:.6f} != expected {EXPECT_FULL}/{EXPECT_TAIL} " \
        f"(trunk mis-load or wrong eval protocol -- fix before training)"

    lens = np.array([len(u["items"]) for u in users]); order = np.argsort(lens)
    batches_all = make_batches(users, order)
    anneal_steps = args.anneal_steps if args.anneal_steps > 0 else len(batches_all)   # default: 1 epoch 0->beta_max
    gstep = start_ep * len(batches_all)
    log(f"[train] {len(batches_all)} adaptive batches/epoch | beta 0->{args.beta_max} over {anneal_steps} steps")
    Sgates = [a * float(np.mean(sig0)) for a in (0.25, 1.0, 4.0)] if args.gate_s <= 0 else [args.gate_s]

    torch.manual_seed(1234)
    canary_done = start_ep > 0
    for ep in range(start_ep, args.epochs):
        vh.train(); rng = np.random.default_rng(100 + ep)
        batches = list(batches_all); rng.shuffle(batches)
        t_ep = time.time(); run_n = 0.0; run_k = 0.0; nb = 0
        for bat in batches:
            exs = [e for e in (make_pb_example(users[i], KT, VT, rng) for i in bat) if e is not None]
            if not exs:
                continue
            L = max(len(e[0]) for e in exs); B = len(exs)
            ids = np.zeros((B, L), np.int64); vals = np.zeros((B, L), np.float32)
            lv = np.zeros((B, L), np.int64); pad = np.ones((B, L), bool)
            tgt = torch.zeros((B, ni), dtype=torch.float32)
            for r, (tid, sv, l_, k_, tg) in enumerate(exs):                 # kn unused: pbC has nknow=0
                n_ = len(tid)
                ids[r, :n_] = tid; vals[r, :n_] = sv; lv[r, :n_] = l_; pad[r, :n_] = False
                tgt[r, tg] = 1.0
            mu, rpool = encode_pool(enc, torch.from_numpy(ids), torch.from_numpy(vals),
                                    torch.from_numpy(pad), torch.from_numpy(lv))    # frozen, no grad
            V, logd = vh(rpool)                                                     # grads -> W_V, W_d only
            Dm = F.softplus(logd) + DFLOOR                                          # floored (matches kl_lowrank)
            eps1 = torch.randn(B, RANK); eps2 = torch.randn(B, D)                   # 1 MC sample (flagged)
            zp = mu + torch.einsum("bdr,br->bd", V, eps1) + torch.sqrt(Dm) * eps2   # mu literally frozen
            logits = zp @ Wd.T + bd                                                 # REAL decoder + LEARNED bias
            nll = -((F.log_softmax(logits, -1) * tgt).sum(-1) / tgt.sum(-1).clamp_min(1.0)).mean()
            beta = args.beta_max * min(1.0, gstep / max(anneal_steps, 1))
            kl = kl_lowrank(V, logd, mu, enc.z0.detach(), vh).mean()
            loss = nll + beta * kl
            opt.zero_grad(); loss.backward(); opt.step()
            gstep += 1; run_n += float(nll); run_k += float(kl); nb += 1
            if not canary_done:                                                     # step-1 freeze canary
                dr = trunk_drift(enc, decoder, SNAP)
                assert dr == 0.0, f"FREEZE BROKEN after 1 step: trunk drift {dr:.2e}"
                log(f"[canary] step-1 trunk+decoder drift = {dr:.2e} (PASS)")
                canary_done = True
            if nb % 50 == 0:
                log(f"  [vh] ep{ep} b{nb}/{len(batches)} NLL={run_n / nb:.4f} KL={run_k / nb:.2f} "
                    f"beta={beta:.3f} {(time.time() - t_ep) / 60:.1f}m")

        # ---------- per-epoch gates ----------
        vh.eval()
        dr = trunk_drift(enc, decoder, SNAP)
        f, t = eval_student(enc, None, base, SPLv, vusers, Wd, bd)
        g0 = (dr == 0.0) and (f == f0) and (t == t0)
        log(f"[G0 ep{ep + 1}] drift={dr:.2e} mu-eval full={f:.6f} tail={t:.6f} "
            f"bit-identical={'PASS' if g0 else 'FAIL'}")
        assert dr == 0.0, "FREEZE BROKE mid-training"
        assert f == f0 and t == t0, "G0 FAIL: frozen-mean eval moved (eval must be bit-identical)"
        g1 = gate_g1(enc, vh, decoder, base, recs, KV, VV, ni)
        g2 = gate_g2(enc, vh, decoder, base, recs, KV, VV, ni, Sgates) \
            if (ep + 1) % args.g2_every == 0 or ep + 1 == args.epochs else {"skipped": True}
        blob = {"varhead": vh.state_dict(), "opt": opt.state_dict(), "epoch": ep + 1,
                "base": args.base, "full": f, "tail": t, "g1": g1, "g2": g2,
                "nll": run_n / max(nb, 1), "kl": run_k / max(nb, 1)}
        safe_save(blob, ckp)
        safe_save(blob, os.path.join(OUT, f"{args.tag}_ep{ep + 1}.pt"))              # save-all rule
        log(f"[vh ep{ep + 1}] NLL={run_n / max(nb, 1):.4f} KL={run_k / max(nb, 1):.2f} "
            f"({(time.time() - t_ep) / 60:.1f}m) saved {args.tag}_ep{ep + 1}.pt")
    log("[vh] done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="pbC_best")        # frozen trunk checkpoint stem under .cache/set_mn/
    ap.add_argument("--tag", default="vhead")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--beta_max", type=float, default=1.0)
    ap.add_argument("--anneal_steps", type=int, default=0)   # 0 => one full epoch of linear 0->beta_max
    ap.add_argument("--gate_s", type=float, default=0.0)     # 0 => grid {0.25,1,4} x mean(diag Sigma0)
    ap.add_argument("--g2_every", type=int, default=1)       # G2 is decisive: run every epoch by default
    ap.add_argument("--resume", action="store_true")
    main(ap.parse_args())
