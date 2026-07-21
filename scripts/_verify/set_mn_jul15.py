"""set_mn.py -- SET-input encoder + MULTINOMIAL decoder. Contract: casper/ARCH_SET_MULTINOMIAL.md.

PHASE A (this file, cmd_pa): does a SELF-ATTENTION set-encoder DISTILL to a0c on FULL profiles and reach
full-profile NDCG@10 >= 0.486? The make-or-break architecture gate (prove it can be a SOTA recommender).
Teacher = a0c SignedAE (frozen). Student = transformer over item tokens (item_emb init a0c factors + continuous
value) -> CLS -> z. Loss = MSE(z_student, z_a0c). Decoder = a0c (frozen) for the NDCG eval.

HARD RULE #1: NO token caps / NO profile truncation ever -> length-bucketed micro-batching. 300 study quarantined.
"""
import os, sys, time, argparse, math
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count()))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(os.cpu_count())
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "instrument2")); sys.path.insert(0, _HERE)
import signed_latent as SL
from signed_latent import (load_arena_base, build_splits, cohort, ndcg10, safe_save, scale_rating,
                           build_train_users, SEEDS, LO, HI, SignedAE, log)

A0C = "C:/dev/phd/casper/.cache/signed_latent/a0c_best.pt"
OUT = "C:/dev/phd/casper/.cache/set_mn"; os.makedirs(OUT, exist_ok=True)
D = 512


class MAB(nn.Module):
    """Multihead attention block (Set Transformer): MAB(Q,K) = LN(H + FF(H)), H = LN(Q + Attn(Q,K,K))."""
    def __init__(self, d, nhead, drop=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, nhead, dropout=drop, batch_first=True)
        self.ln0 = nn.LayerNorm(d); self.ln1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * d, d))

    def forward(self, Q, K, key_padding_mask=None):
        h, _ = self.attn(Q, K, K, key_padding_mask=key_padding_mask, need_weights=False)
        h = self.ln0(Q + h)
        return self.ln1(h + self.ff(h))


class ISAB(nn.Module):
    """Induced Set Attention Block: tokens interact GLOBALLY through m learned inducing points.
    O(L*m) instead of O(L^2) -> handles the 14,040-item profiles. Permutation-invariant, arbitrary L."""
    def __init__(self, d, nhead, m=32, drop=0.1):
        super().__init__()
        self.I = nn.Parameter(torch.randn(1, m, d) * 0.02)
        self.mab0 = MAB(d, nhead, drop)      # inducing points attend to the set
        self.mab1 = MAB(d, nhead, drop)      # the set attends back to the inducing points

    def forward(self, X, pad):
        b = X.shape[0]
        H = self.mab0(self.I.expand(b, -1, -1), X, key_padding_mask=pad)   # (B, m, d)
        return self.mab1(X, H)                                             # (B, L, d)


class PMA(nn.Module):
    """Pooling by multihead attention: a learned seed attends over the set -> one vector."""
    def __init__(self, d, nhead, drop=0.1):
        super().__init__()
        self.S = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.mab = MAB(d, nhead, drop)

    def forward(self, X, pad):
        b = X.shape[0]
        return self.mab(self.S.expand(b, -1, -1), X, key_padding_mask=pad)[:, 0]


class SetEncoder(nn.Module):
    """FAST set-encoder: tokens -> attend INTO m inducing points -> inducing points self-attend (global
    interaction) -> pool -> z. Permutation-invariant, ANY set length, real cross-token interaction, but NO
    per-token FFN (we only need a pooled belief) -> ~10-20x faster than full ISAB on 14k-token profiles."""
    def __init__(self, ni, d=D, nhead=4, m=32, nlayers=2, token_mode="mlp", pool="attn", nlev=10, nknow=0):
        super().__init__()
        self.ni = ni; self.d = d; self.token_mode = token_mode; self.pool = pool
        self.item_emb = nn.Embedding(ni, d)                        # init from a0c decoder factors
        self.tok_mlp = nn.Sequential(nn.Linear(d + 1, d), nn.GELU(), nn.Linear(d, d))
        # FILM token: learned per-RATING scale+shift tables (10 discrete half-star levels).
        # init gamma=1, beta~small  ->  token = item_emb + beta[r]  = the PURE ADDITIVE token (author's null).
        # The model can LEARN gamma negative (dislike-as-negation) if the data asks; nothing imposed.
        # ~100x cheaper than tok_mlp (2 lookups + elementwise vs a 0.5M-FLOP MLP per token).
        self.gamma = nn.Embedding(nlev, d); self.beta = nn.Embedding(nlev, d)
        nn.init.ones_(self.gamma.weight); nn.init.normal_(self.beta.weight, std=0.02)
        # KNOWLEDGE = a SEPARATE learned signal, ADDED to the token. NEVER multiplied into value (author rule).
        # init ZERO => additive null: at start the knowledge level changes nothing; the model learns what it means.
        self.know_emb = nn.Embedding(nknow, d) if nknow else None
        if self.know_emb is not None:
            nn.init.zeros_(self.know_emb.weight)
        self.I = nn.Parameter(torch.randn(1, m, d) * 0.02)         # inducing points
        self.mab_in = MAB(d, nhead)                                # I attends to the set  (O(L*m), FF on m only)
        self.sab = nn.ModuleList([MAB(d, nhead) for _ in range(nlayers)])   # interaction among inducing points
        self.pma = PMA(d, nhead)                                   # pool -> one vector
        self.z0 = nn.Parameter(torch.zeros(d))                     # empty-set prior = PRIOR MEAN
        # ---- BELIEF pooling (conjugate-Gaussian): posterior mean = precision-weighted evidence + prior ----
        # lambda_t = LEARNED per-token precision (confidence), fed the ISAB context so it can DISCOUNT
        # redundant/collinear evidence instead of double-counting it (the whitening cure).
        # NOTE z0=0 + a0c-warm-started decoder => empty set decodes to the decoder BIAS = popularity. Exact.
        self.lam_head = nn.Linear(2 * d, 1)                        # scalar precision (diag would re-add the d->d MLP)
        self.log_p0 = nn.Parameter(torch.zeros(d))                 # prior precision (diagonal, LEARNED)
        self.pscale = nn.Parameter(torch.ones(d))                  # scale evidence to the decoder's norm;
        self.last_prec = None                                      # z0 + s*(mu-z0) keeps the intercept EXACT
        self.head = nn.Linear(d + 1, d)                            # + log(1+m) set-size feature

    def forward(self, ids, vals, pad, lvs=None, kn=None):
        b = ids.shape[0]
        e = self.item_emb(ids)
        if self.token_mode == "film":
            x = self.gamma(lvs) * e + self.beta(lvs)                                     # (B,L,d)  CHEAP
            if self.know_emb is not None and kn is not None:
                x = x + self.know_emb(kn)                       # SEPARATE + ADDITIVE. Never multiplied.
        else:
            x = self.tok_mlp(torch.cat([e, vals.unsqueeze(-1)], dim=-1))                 # (B,L,d)
        h = self.mab_in(self.I.expand(b, -1, -1), x, key_padding_mask=pad)              # (B,m,d)
        for blk in self.sab:
            h = blk(h, h)                                                               # global interaction
        p = self.pma(h, None)                                                           # (B,d) context
        p = torch.nan_to_num(p)          # all-padded row => softmax over -inf => NaN; empty set must give z0
        if self.pool == "belief":
            keep = (~pad).unsqueeze(-1).float()                                         # (B,L,1)
            ctx = p.unsqueeze(1).expand(-1, x.shape[1], -1)                             # (B,L,d)
            lam = F.softplus(self.lam_head(torch.cat([x, ctx], dim=-1))) * keep         # (B,L,1) >=0, pad->0
            p0 = F.softplus(self.log_p0)                                                # (d,) prior precision
            num = p0 * self.z0 + (lam * x).sum(1)                                       # (B,d)
            den = p0 + lam.sum(1)                                                       # (B,d) broadcast
            self.last_prec = den                                                        # belief precision (for entropy)
            mu = num / den                                                              # posterior mean; n=0 -> z0
            return self.z0 + self.pscale * (mu - self.z0)                               # empty set -> z0 EXACTLY
        sz = (~pad).sum(-1, keepdim=True).float().clamp_min(1.0).log1p()
        return self.z0 + self.head(torch.cat([p, sz], dim=-1))


def sv_to_level(sv):
    """signed value -> discrete half-star rating level 0..9 (scale_rating: star = sv*2.25 + 2.75)."""
    star = np.asarray(sv, np.float64) * 2.25 + 2.75
    return np.clip(np.rint(star * 2).astype(np.int64) - 1, 0, 9)


def load_teacher(ni):
    t = SignedAE(ni, use_mask=True)
    blob = torch.load(A0C, map_location="cpu"); t.load_state_dict(blob["model"]); t.eval()
    for p in t.parameters():
        p.requires_grad_(False)
    return t


ATTN_BUDGET = 1_000_000        # cap on B * L (ISAB is O(L*m) -> LINEAR in L). NOT a data cap: all users, all items,
MAX_B = 256 # we only vary HOW MANY users share a batch. HARD RULE #1 intact.


def make_batches(users, order):
    """Length-bucketed ADAPTIVE batching: fewer users per batch when profiles are long, so B*L^2 stays bounded.
    Never truncates a profile, never drops a user."""
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


def pad_batch(users, idxs):
    """Length-bucketed padded (ids, vals, pad) from users' rated items (all of them; NO cap)."""
    L = max(len(users[i]["items"]) for i in idxs)
    B = len(idxs)
    ids = np.zeros((B, L), np.int64); vals = np.zeros((B, L), np.float32); pad = np.ones((B, L), bool)
    for r, i in enumerate(idxs):
        it = users[i]["items"]; sv = users[i]["sv"]; k = len(it)
        ids[r, :k] = it; vals[r, :k] = sv; pad[r, :k] = False
    return torch.from_numpy(ids), torch.from_numpy(vals), torch.from_numpy(pad)


def eval_student(student, teacher, base, SPL, users, decoder_W, decoder_b):
    """Full-profile NDCG@10 via decoder(student_z). revealed=liked_prof (a0c 'full' protocol)."""
    student.eval(); ni = base["ni"]; headmask = base["headmask"]; ff, tt = [], []
    recs = []
    for u in users:
        profset, held, prof_r, held_r = SPL[u]
        tlike = [j for j in held if held_r[j] >= LO]
        if not tlike:
            continue
        liked = [j for j in prof_r if prof_r[j] >= LO]
        if not liked:
            continue
        recs.append((np.array(liked, np.int64),
                     scale_rating(np.array([prof_r[j] for j in liked], np.float64)).astype(np.float32),
                     profset, tlike))
    if not recs:
        return float("nan"), float("nan")
    with torch.no_grad():
        for b in range(0, len(recs), 128):
            chunk = recs[b:b + 128]
            L = max(len(r[0]) for r in chunk); B = len(chunk)
            ids = np.zeros((B, L), np.int64); vals = np.zeros((B, L), np.float32); pad = np.ones((B, L), bool)
            lvs = np.zeros((B, L), np.int64)
            for r, (it, sv, _, _) in enumerate(chunk):
                ids[r, :len(it)] = it; vals[r, :len(it)] = sv; pad[r, :len(it)] = False
                lvs[r, :len(it)] = sv_to_level(sv)
            z = student(torch.from_numpy(ids), torch.from_numpy(vals), torch.from_numpy(pad),
                        torch.from_numpy(lvs))
            sc = (z @ decoder_W.T + decoder_b).numpy().astype(np.float64)
            for r, (_, _, profset, tlike) in enumerate(chunk):
                nf = ndcg10(sc[r], tlike, profset, headmask, False)
                nt = ndcg10(sc[r], tlike, profset, headmask, True)
                if nf is not None: ff.append(nf)
                if nt is not None: tt.append(nt)
    return (float(np.mean(ff)) if ff else float("nan"), float(np.mean(tt)) if tt else float("nan"))


def make_input_target(u, rng, drop_max=0.5):
    """a0c's own curriculum: input = profile subset (random dropout), target = liked NOT in input (leak-free)."""
    its = u["items"]; n = len(its)
    keep = rng.random(n) >= rng.uniform(0.0, drop_max)
    if not keep.any():
        keep[rng.integers(0, n)] = True
    inp = its[keep]; sv = u["sv"][keep]
    tgt = np.setdiff1d(u["liked"], inp, assume_unique=False)
    if len(tgt) == 0:
        return None
    return inp, sv, tgt


# ============================ PHASE B — CONCEPTS + ATTRIBUTES ============================
# ONE INDEX SPACE (the pre-VAE recipe that worked): a concept is LITERALLY another entity id in the SAME
# embedding table.  items [0, ni)  |  concepts+entities [ni, ni+NC).
# LEVELS (15): 0-9 item half-star ; 10-13 concept ordinal (hated/meh/liked/loved) ; 14 = REFUSAL.
# KNOWLEDGE (3): no_clue / rough_idea / know_well -- a SEPARATE ADDITIVE embedding, never multiplied.
# PROVEN TWICE ALREADY (u1: concept-only cold k32=0.304 > intercept 0.2551 ; pre-VAE: attr-only 0.395->0.51).
# THE HONEST NEW PART: this tests the recipe against the REALISTIC ANSWERER, not oracle member-means.
NC = 1628
NLEV = 15
LV_REFUSE = 14
RSD = "C:/dev/phd/casper/.cache/rich_signal"


def load_answerer(tag):
    K = np.load(RSD + "/mm_" + tag + "_know.npy")
    V = np.load(RSD + "/mm_" + tag + "_val.npy")
    uids = np.load(RSD + "/mm_" + tag + "_uids.npy")
    return {int(u): r for r, u in enumerate(uids)}, K, V


def concept_answers(row, K, V, cids):
    """Answers from the DISTILLED ANSWERER. A REFUSAL IS AN EXPLICIT TOKEN, NOT AN ABSENCE -- it burns the
    turn. (A policy allowed to dodge converges to the 94%-answerable equilibrium that killed E0.)"""
    kn = K[row, cids].astype(np.int64)
    vl = V[row, cids].astype(np.int64)
    ans = (kn >= 1) & (vl >= 0)
    lv = np.where(ans, 10 + np.clip(vl, 0, 3), LV_REFUSE).astype(np.int64)
    kk = np.where(ans, np.clip(kn, 0, 2), 0).astype(np.int64)
    return lv, kk


def build_pb_users(base, umap):
    """build_train_users + the uid (needed to index the answerer tables). Full data, no cap."""
    tra_u = base["tra_u"]; tra_i = base["tra_i"]; tra_r = base["tra_r"]
    o = np.argsort(tra_u, kind="stable")
    tu, ti, tr = tra_u[o], tra_i[o], tra_r[o]
    users = []
    N = len(tu); b = 0
    while b < N:
        e = b
        while e < N and tu[e] == tu[b]:
            e += 1
        uid = int(tu[b])
        row = umap.get(uid)
        if row is not None:
            its = ti[b:e].astype(np.int64); rs = tr[b:e].astype(np.float64)
            srt = np.argsort(its); its = its[srt]; rs = rs[srt]
            liked = its[rs >= LO]
            if len(liked) >= 2:
                users.append(dict(items=its, sv=scale_rating(rs).astype(np.float32),
                                  liked=liked, row=row))
        b = e
    return users


def make_pb_example(u, K, V, rng):
    """CONCEPT-FORCING CURRICULUM (the pre-VAE recipe: ATTR_REVEAL_P=0.5 -- half of all reveals are
    concept-only, so concepts MUST carry the belief).
      item-only (0.35) : strength preservation, so full-profile accuracy does not rot
      CONCEPT-ONLY (0.35): forcing -- the belief must be built from concepts alone
      mixed (0.30)     : the realistic interview
    Target = liked items NOT revealed (leak-free)."""
    r = rng.random()
    its = u["items"]; n = len(its)
    tid = []; sv = []; lv = []; kk = []
    if r < 0.65:                                                   # items present (item-only or mixed)
        keep = rng.random(n) >= rng.uniform(0.0, 0.9)              # heavy dropout => short interviews too
        if not keep.any():
            keep[rng.integers(0, n)] = True
        ii_ = its[keep]; ss = u["sv"][keep]
        tid.append(ii_); sv.append(ss)
        lv.append(sv_to_level(ss)); kk.append(np.full(len(ii_), 2, np.int64))   # rated => know_well
    if r >= 0.35:                                                  # concepts present (concept-only or mixed)
        k = int(rng.integers(1, 33))                               # log-uniform-ish 1..32 concept reveals
        cids = rng.choice(NC, size=k, replace=False)
        clv, ckk = concept_answers(u["row"], K, V, cids)
        ni_ = K.shape[1]                                           # unused; keep explicit
        tid.append(cids.astype(np.int64) + PB_NI[0])               # concept token id = ni + c
        sv.append(np.zeros(k, np.float32))                         # value lives in the LEVEL, not the scalar
        lv.append(clv); kk.append(ckk)
    if not tid:
        return None
    tid = np.concatenate(tid); sv = np.concatenate(sv)
    lv = np.concatenate(lv); kk = np.concatenate(kk)
    revealed = tid[tid < PB_NI[0]]
    tgt = np.setdiff1d(u["liked"], revealed, assume_unique=False)   # leak-free
    if len(tgt) == 0:
        return None
    return tid, sv.astype(np.float32), lv, kk, tgt


PB_NI = [0]        # set to ni at startup (module-level so make_pb_example can see it)


def eval_concept_only(student, base, SPL, users, Wd, bd, ni, umapV, KV, VV, ks=(1, 2, 4, 8, 16, 32)):
    """CONCEPT-ONLY COLD-START (the gate that the additive operator FAILED: 0.226 vs intercept 0.2551).
    Concepts are sampled at RANDOM from the bank and answered by the REALISTIC ANSWERER -- refusals included,
    and a refusal BURNS THE TURN. No cherry-picking answerable concepts."""
    student.eval(); headmask = base["headmask"]
    rng = np.random.default_rng(7)
    out = {}
    recs = [(u,) + SPL[u] for u in users if umapV.get(u) is not None]
    with torch.no_grad():
        # intercept: the empty set
        z = student(torch.zeros((1, 1), dtype=torch.long), torch.zeros((1, 1)),
                    torch.ones((1, 1), dtype=torch.bool), torch.zeros((1, 1), dtype=torch.long),
                    torch.zeros((1, 1), dtype=torch.long))
        sc0 = (z @ Wd.T + bd).numpy().astype(np.float64)[0]
        ff = []
        for u, profset, held, prof_r, held_r in recs:
            tl = [j for j in held if held_r[j] >= LO]
            if tl:
                v = ndcg10(sc0, tl, profset, headmask, False)
                if v is not None:
                    ff.append(v)
        out["intercept"] = round(float(np.mean(ff)), 4) if ff else float("nan")

        for k in ks:
            ff = []
            for b in range(0, len(recs), 256):
                ch = recs[b:b + 256]
                B = len(ch)
                ids = np.zeros((B, k), np.int64); lv = np.zeros((B, k), np.int64)
                kk = np.zeros((B, k), np.int64)
                for r, (u, profset, held, prof_r, held_r) in enumerate(ch):
                    cids = rng.choice(NC, size=k, replace=False)
                    a, c = concept_answers(umapV[u], KV, VV, cids)
                    ids[r] = cids + ni; lv[r] = a; kk[r] = c
                z = student(torch.from_numpy(ids), torch.zeros((B, k)),
                            torch.zeros((B, k), dtype=torch.bool),
                            torch.from_numpy(lv), torch.from_numpy(kk))
                sc = (z @ Wd.T + bd).numpy().astype(np.float64)
                for r, (u, profset, held, prof_r, held_r) in enumerate(ch):
                    tl = [j for j in held if held_r[j] >= LO]
                    if tl:
                        v = ndcg10(sc[r], tl, profset, headmask, False)
                        if v is not None:
                            ff.append(v)
            out["k%d" % k] = round(float(np.mean(ff)), 4) if ff else float("nan")
    student.train()
    return out


def cmd_pb(args):
    """PHASE B. DECODER FROZEN (author's staging): concepts must learn to SPEAK the decoder's language,
    not reshape it."""
    base = load_arena_base(); ni = base["ni"]; NT = ni + NC
    PB_NI[0] = ni
    ck = torch.load(os.path.join(OUT, args.base + ".pt"), map_location="cpu")
    # AUTO-DETECT the pool from the base checkpoint. A belief checkpoint carries lam_head/log_p0/pscale; an
    # attention encoder would silently IGNORE them and we would train a different model than we think.
    pool = "belief" if any(k.startswith(("lam_head", "log_p0", "pscale")) for k in ck["student"]) else "attn"
    if pool != args.pool:
        log("[pb] pool AUTO-DETECTED from %s: %s (flag said %s -- using the checkpoint)"
            % (args.base, pool, args.pool))
    log("[pb] base=%s (full-profile %.4f) pool=%s%s"
        % (args.base, ck.get("full", float("nan")), pool,
           "  [BELIEF: z carries a precision]" if pool == "belief" else "  [point estimate z only]"))

    student = SetEncoder(NT, token_mode="film", pool=pool, nlev=NLEV, nknow=3)
    decoder = nn.Linear(D, ni)
    sd = ck["student"]
    with torch.no_grad():
        student.item_emb.weight[:ni].copy_(sd["item_emb.weight"])
        student.gamma.weight[:10].copy_(sd["gamma.weight"])
        student.beta.weight[:10].copy_(sd["beta.weight"])
        named = dict(student.named_parameters())
        for kk_, vv_ in sd.items():
            if kk_.startswith(("item_emb", "gamma", "beta", "know_emb")):
                continue
            if kk_ in named and named[kk_].shape == vv_.shape:
                named[kk_].copy_(vv_)
        decoder.load_state_dict(ck["decoder"])
        # CONCEPT WARM-START: pop-weighted MEMBER-BAG mean of the item embeddings (encoder-faithful init).
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from reconciled import ConceptBank
        cb = ConceptBank(ni, base["cnt"])
        M = cb.Mw.tocsr() if hasattr(cb.Mw, 'tocsr') else cb.Mw    # pop-weighted member bag
        rs = np.asarray(M.sum(1)).ravel(); rs[rs == 0] = 1.0
        E = (M @ sd["item_emb.weight"].numpy()) / rs[:, None]
        student.item_emb.weight[ni:ni + M.shape[0]].copy_(torch.from_numpy(E).float())
        log("[pb] concept rows warm-started from member-bags: %d rows, nnz %d" % (M.shape[0], M.nnz))
    for p_ in decoder.parameters():
        p_.requires_grad_(False)                                   # ***** DECODER FROZEN *****
    Wd = decoder.weight.detach(); bd = decoder.bias.detach()

    opt = torch.optim.AdamW(student.parameters(), lr=3e-4, weight_decay=1e-4)
    umap, KT, VT = load_answerer("train")
    umapV, KV, VV = load_answerer("val")
    users = build_pb_users(base, umap)
    SPLv = build_splits(base, SEEDS[0]); vusers = cohort(base, SPLv, "val")
    lens = np.array([len(u["items"]) for u in users]); order = np.argsort(lens)
    batches_all = make_batches(users, order)
    log("[pb] %d train users w/ answerer rows | tokens %d (items %d + concepts %d) | DECODER FROZEN | pool=%s"
        % (len(users), NT, ni, NC, pool))
    log("[pb] %d adaptive batches" % len(batches_all))

    start_ep = 0; best = -1.0; bad = 0
    ckp = os.path.join(OUT, args.tag + ".pt")
    if args.resume and os.path.exists(ckp):
        blob = torch.load(ckp, map_location="cpu")
        student.load_state_dict(blob["student"]); start_ep = blob.get("epoch", 0)
        best = blob.get("full", -1.0)
        if "opt" in blob:
            opt.load_state_dict(blob["opt"])
        log("[pb] RESUMED ep%d (full=%.4f)" % (start_ep, best))

    f0, t0 = eval_student(student, None, base, SPLv, vusers, Wd, bd)
    cc0 = eval_concept_only(student, base, SPLv, vusers, Wd, bd, ni, umapV, KV, VV)
    log("[pb] INIT full=%.4f tail=%.4f (Phase-A base %.4f) | CONCEPT-ONLY %s"
        % (f0, t0, ck.get("full", 0.0), cc0))

    for ep in range(start_ep, args.epochs):
        student.train(); rng = np.random.default_rng(100 + ep)
        batches = list(batches_all); rng.shuffle(batches)
        t_ep = time.time(); run = 0.0; nb = 0
        for bat in batches:
            exs = [e for e in (make_pb_example(users[i], KT, VT, rng) for i in bat) if e is not None]
            if not exs:
                continue
            L = max(len(e[0]) for e in exs); B = len(exs)
            ids = np.zeros((B, L), np.int64); vals = np.zeros((B, L), np.float32)
            lv = np.zeros((B, L), np.int64); kk = np.zeros((B, L), np.int64)
            pad = np.ones((B, L), bool)
            tgt = torch.zeros((B, ni), dtype=torch.float32)
            for r, (tid, sv, l_, k_, tg) in enumerate(exs):
                n_ = len(tid)
                ids[r, :n_] = tid; vals[r, :n_] = sv; lv[r, :n_] = l_; kk[r, :n_] = k_
                pad[r, :n_] = False
                tgt[r, tg] = 1.0
            z = student(torch.from_numpy(ids), torch.from_numpy(vals), torch.from_numpy(pad),
                        torch.from_numpy(lv), torch.from_numpy(kk))
            logits = z @ Wd.T + bd
            nll = -((F.log_softmax(logits, -1) * tgt).sum(-1) / tgt.sum(-1).clamp_min(1.0)).mean()
            opt.zero_grad(); nll.backward(); opt.step()
            run += float(nll); nb += 1
            if nb % 50 == 0:
                log("  [pb] ep%d b%d/%d NLL=%.4f %.1fm" % (ep, nb, len(batches), run / nb,
                                                           (time.time() - t_ep) / 60))
        f, t = eval_student(student, None, base, SPLv, vusers, Wd, bd)
        cc = eval_concept_only(student, base, SPLv, vusers, Wd, bd, ni, umapV, KV, VV)
        log("[pb ep%d] NLL=%.4f full=%.4f tail=%.4f (%s) | CONCEPT-ONLY %s (%.1fm)"
            % (ep + 1, run / max(nb, 1), f, t,
               "HOLDS >= base" if f >= args.gate else "BELOW base",
               cc, (time.time() - t_ep) / 60))
        blob_ep = {"student": student.state_dict(), "decoder": decoder.state_dict(),
                   "opt": opt.state_dict(), "epoch": ep + 1, "full": f, "tail": t}
        safe_save(blob_ep, ckp)
        safe_save(blob_ep, os.path.join(OUT, "%s_ep%d.pt" % (args.tag, ep + 1)))  # keep EVERY epoch (eval is broken; pick later)
        if f > best:
            best = f; bad = 0
            safe_save({"student": student.state_dict(), "decoder": decoder.state_dict(),
                       "epoch": ep + 1, "full": f, "tail": t}, os.path.join(OUT, args.tag + "_best.pt"))
        else:
            bad += 1
            if bad >= args.patience:
                log("[pb] CONVERGED"); break
    log("[pb] done best full-profile=%.4f" % best)


def cmd_pa(args):
    """FORCED Phase A: train the set-encoder on the REAL objective (multinomial NLL of held-liked, a0c's own
    curriculum) with z-distillation demoted to an AUX anchor, decoder co-trained from a0c warm-start.
    Pure z-mimicry caps at mimicry quality; the gate is NDCG >= 0.486, so optimize NDCG directly."""
    base = load_arena_base(); ni = base["ni"]
    teacher = load_teacher(ni)
    student = SetEncoder(ni, token_mode=args.token, pool=args.pool)
    decoder = nn.Linear(D, ni)                                                        # co-trained, a0c warm-start
    with torch.no_grad():
        student.item_emb.weight.copy_(teacher.decoder.weight.detach())                # item_emb = a0c factors
        decoder.weight.copy_(teacher.decoder.weight.detach())
        decoder.bias.copy_(teacher.decoder.bias.detach())
    Wd = decoder.weight; bd = decoder.bias
    LAM = args.lam                                                                    # z-distill aux weight
    opt = torch.optim.AdamW(list(student.parameters()) + list(decoder.parameters()),
                            lr=3e-4, weight_decay=1e-4)
    users = build_train_users(base)                                                   # items + sv (signed value)
    log(f"[pa] {len(users)} train users; teacher a0c frozen; student set-encoder (d={D}, token={args.token}, pool={args.pool})")
    SPLv = build_splits(base, SEEDS[0]); vusers = cohort(base, SPLv, "val")
    # sort by length for bucketed ADAPTIVE batching (HARD RULE #1: bucket, never cap)
    lens = np.array([len(u["items"]) for u in users])
    order = np.argsort(lens)
    batches_all = make_batches(users, order)
    log(f"[pa] profiles: min={lens.min()} med={int(np.median(lens))} max={lens.max()}; "
        f"{len(batches_all)} adaptive batches (B*L<={ATTN_BUDGET//1_000_000}M, ISAB O(L*m))")
    start_ep = 0; best = -1.0; bad = 0
    ck = os.path.join(OUT, f"{args.tag}.pt")
    if args.resume and os.path.exists(ck):
        blob = torch.load(ck, map_location="cpu")
        student.load_state_dict(blob["student"]); start_ep = blob.get("epoch", 0)
        decoder.load_state_dict(blob["decoder"])          # co-trained: MUST restore, else silent reset to a0c
        Wd = decoder.weight.detach(); bd = decoder.bias.detach()
        if "opt" in blob:
            opt.load_state_dict(blob["opt"])              # Adam moments: else a loss spike at every restart
        best = blob.get("full", -1.0)
        log(f"[pa] RESUMED from ep{start_ep} (full={best:.4f}) decoder+opt restored")
    f0, t0 = eval_student(student, teacher, base, SPLv, vusers, Wd, bd)
    log(f"[pa] INIT student full-profile NDCG@10 full={f0:.4f} tail={t0:.4f}  (a0c target 0.4961; gate 0.486)")
    for ep in range(start_ep, args.epochs):
        student.train(); rng = np.random.default_rng(ep)
        batches = list(batches_all)
        rng.shuffle(batches)
        t_ep = time.time(); run = 0.0; nb = 0
        for bat in batches:
            # a0c curriculum: input = profile subset, target = liked not in input
            exs = [(i, make_input_target(users[i], rng)) for i in bat]
            exs = [(i, e) for i, e in exs if e is not None]
            if not exs:
                continue
            L = max(len(e[1][0]) for e in exs); B = len(exs)
            ids = np.zeros((B, L), np.int64); vals = np.zeros((B, L), np.float32); pad = np.ones((B, L), bool)
            lvs = np.zeros((B, L), np.int64)
            xv = torch.zeros((B, ni), dtype=torch.float32); tgt = torch.zeros((B, ni), dtype=torch.float32)
            for r, (i, (inp, sv, tg)) in enumerate(exs):
                k = len(inp)
                ids[r, :k] = inp; vals[r, :k] = sv; pad[r, :k] = False; lvs[r, :k] = sv_to_level(sv)
                if LAM > 0:
                    xv[r, inp] = torch.from_numpy(sv)                   # teacher sees the SAME input
                tgt[r, tg] = 1.0
            if LAM > 0:
                with torch.no_grad():
                    z_t = teacher.encode(xv)
            z_s = student(torch.from_numpy(ids), torch.from_numpy(vals), torch.from_numpy(pad),
                          torch.from_numpy(lvs))
            logits = z_s @ Wd.T + bd
            logsm = F.log_softmax(logits, dim=-1)
            nll = -((logsm * tgt).sum(-1) / tgt.sum(-1).clamp_min(1.0)).mean()        # THE objective
            loss = nll
            if LAM > 0:                                                               # anchor OFF at lam=0:
                loss = loss + LAM * F.mse_loss(z_s, z_t)                              # no ceiling, no teacher pass
            opt.zero_grad(); loss.backward(); opt.step()
            run += float(nll); nb += 1
            if nb % 50 == 0:
                log(f"  [pa] ep{ep} b{nb}/{len(batches)} NLL={run/nb:.4f} {(time.time()-t_ep)/60:.1f}m")
        f, t = eval_student(student, teacher, base, SPLv, vusers, Wd, bd)
        log(f"[pa ep{ep+1}] NLL={run/max(nb,1):.4f} full-profile NDCG@10 full={f:.4f} tail={t:.4f} "
            f"({'PASS >=0.486' if f>=0.486 else 'below gate'}) ({(time.time()-t_ep)/60:.1f}m)")
        safe_save({"student": student.state_dict(), "decoder": decoder.state_dict(), "opt": opt.state_dict(),
                   "epoch": ep + 1, "full": f, "tail": t},
                  os.path.join(OUT, f"{args.tag}.pt"))
        if f > best:
            best = f; bad = 0
            safe_save({"student": student.state_dict(), "decoder": decoder.state_dict(), "epoch": ep + 1, "full": f, "tail": t},
                      os.path.join(OUT, f"{args.tag}_best.pt"))
        else:
            bad += 1
            log(f"[pa] no improvement over {best:.4f} ({bad}/{args.patience})")
            if bad >= args.patience:
                log(f"[pa] CONVERGED (val NDCG flat for {args.patience} epochs)"); break
    log(f"[pa] done best full-profile={best:.4f} (gate 0.486; a0c 0.4961)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pa", "pb"])
    ap.add_argument("--base", default="pb2_best")
    ap.add_argument("--gate", type=float, default=0.4852)
    ap.add_argument("--tag", default="pa1"); ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--resume", action="store_true"); ap.add_argument("--patience", type=int, default=3); ap.add_argument("--lam", type=float, default=0.1); ap.add_argument("--token", choices=["mlp","film"], default="mlp"); ap.add_argument("--pool", choices=["attn","belief"], default="attn")
    a = ap.parse_args()
    if a.cmd == "pa":
        cmd_pa(a)
    elif a.cmd == "pb":
        cmd_pb(a)
