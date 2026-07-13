"""reconciled.py -- RECONCILED RECOMMENDER: strong item tower + DEDICATED LEARNABLE concept channel.

Contract: casper/DESIGN_RECONCILED.md (Fable-hardened, sec 7 amendments). Three-phase curriculum:
  P1 item pretrain  = a03b_best.pt (signed_latent, full-profile graded items, NDCG@10=0.4961) -- DONE.
  P2 concept warm   = add dedicated concept channel (e_c = ENCODER-FAITHFUL member-bag init; W_shared=I),
                      train on CLEAN emulated interviews (known items + concepts DERIVED from known members,
                      real values, varying item:concept proportions incl. an ITEM-MASKED concept-forcing
                      slice) + a slice of full profiles. Encoder trains too. No refusals, no EASE.
  P3 realistic      = warm P2, LR<=0.1x, distilled-answerer interviews (strategy-chosen, refusals, mehs,
                      EASE-imputed), >=30% full-profile replay, joint peak. [wired after P2 validates]

The concept channel is LOAD-BEARING only in the item-scarce regime; at full item coverage concepts are
correctly redundant (derived from items). The item-masked slice forces gradient into e_c (defeats the
redundancy shortcut). Headline gate = USEFULNESS (knockout/twin early-AUC), NOT the behavioral sweep.

NO DATA CAPS. 173/300 study users QUARANTINED. Leak-free known intersect held = empty. Save all ckpts.
"""
import os, sys, json, time, argparse, math
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count()))
os.environ.setdefault("MKL_NUM_THREADS", str(os.cpu_count()))
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(os.cpu_count())
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "instrument2"))
sys.path.insert(0, _HERE)

# ---- reuse the PROVEN signed_latent machinery (one ruler, identical eval) ----
import signed_latent as SL
from signed_latent import (load_arena_base, build_splits, cohort, ndcg10, safe_save,
                           scale_rating, build_train_users, SEEDS, LO, HI, item_info,
                           STRATEGIES, select_items, SignedEncoder, swish, log, _gumbel_topk)

CACHE = "C:/dev/phd/casper/.cache/instrument2"
TAG_Q = f"{CACHE}/tag_questions.json"
TAG_MEMB = f"{CACHE}/tag_membership.json"
ATTR_BAT = f"{CACHE}/attr_battery_500.json"
A03B = "C:/dev/phd/casper/.cache/signed_latent/a0c_best.pt"   # the STRONG 0.4961 alpha=0 ckpt (NOT a03b=0.4272)
OUTDIR = "C:/dev/phd/casper/.cache/reconciled"
os.makedirs(OUTDIR, exist_ok=True)

# ============================== FORBIDDEN — DO NOT REUSE ==============================
# These two tables are HAND-DRAWN HEURISTICS (author rule: none allowed), and the code below
# MULTIPLIES them (value x confidence) — also forbidden: value, confidence and refusal are
# SEPARATE signals and must never be collapsed into one scalar.
# They survive only inside this SUPERSEDED dense line (u1/SignedAE), which the author rejected on
# 2026-07-13 (a fixed input dim per concept cannot express a continuous/open query).
# The live architecture is scripts/set_mn.py (set input + multinomial decoder), where value is a
# LEARNED per-rating embedding table and confidence/refusal get their own learned slots.
# Do not import, copy, or reintroduce these in any new code path.
VALENCE = np.array([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0], np.float32)   # index by val level 0..3
CONF = np.array([0.0, 0.5, 1.0], np.float32)                          # index by know level 0..2
# ======================================================================================


def star_to_val(star):
    """uncentered mean-member star -> snapped value level {0 hated,1 meh,2 liked,3 loved} (arena_core rule)."""
    return 3 if star >= 4.5 else 2 if star >= 3.5 else 1 if star >= 2.5 else 0


# ======================================================================== concept bank (tags + entities)
class ConceptBank:
    """concepts = 1128 genome tags + 500 entity/attributes = C rows over item space (ni). Pop-weighted
    member-bag per concept (logcnt weighting, head-damped) for the ENCODER-FAITHFUL init. Item index space
    is IDENTICAL to signed_latent (both meta.npz dense ids) -- verified."""
    def __init__(self, ni, cnt):
        self.ni = ni
        rows, cols, wts = [], [], []
        logcnt = np.log(cnt.astype(np.float64) + 1.0)
        names, kinds = [], []
        r = 0
        # ---- tags ----
        tq = json.load(open(TAG_Q))["tags"]
        memb = json.load(open(TAG_MEMB))["membership"]
        for t in tq:
            ms = memb.get(str(t["tagId"]), [])
            m = [j for j in ms if 0 <= j < ni]
            if not m:
                m = []  # keep row (embedding still learnable); init falls back to prior
            for j in m:
                rows.append(r); cols.append(j); wts.append(logcnt[j] + 1e-3)
            names.append(str(t.get("tag", f"tag{t['tagId']}"))); kinds.append("concept")
            r += 1
        self.ntag = r
        # ---- entities ----
        bat = json.load(open(ATTR_BAT))["entities"]
        for e in bat:
            m = [j for j in e.get("member_dense_ids", []) if 0 <= j < ni]
            for j in m:
                rows.append(r); cols.append(j); wts.append(logcnt[j] + 1e-3)
            names.append(str(e.get("name", e.get("entity_id")))); kinds.append("entity")
            r += 1
        self.C = r
        self.nent = self.C - self.ntag
        # pop-weighted membership matrix (C x ni), row-normalized to unit sum (member-bag = weighted mean dir)
        M = sp.csr_matrix((np.asarray(wts, np.float64), (rows, cols)), shape=(self.C, ni))
        rs = np.asarray(M.sum(1)).ravel()
        self.row_sum = rs
        self.has_members = rs > 0
        inv = np.where(rs > 0, 1.0 / rs, 0.0)
        self.Mw = M.multiply(inv[:, None]).tocsr()      # row-normalized pop-weighted membership
        self.Mbin = (M > 0).astype(np.float64).tocsr()  # binary membership (CSR: row=concept -> members)
        self.Mcsc = self.Mbin.tocsc()                   # CSC: column=item -> concepts (fast k-item derivation)
        self.names = names
        self.kinds = np.array(kinds)
        log(f"[conceptbank] C={self.C} (tags={self.ntag} ents={self.nent}), "
            f"with-members={int(self.has_members.sum())}, nnz={M.nnz}")

    def memberbag_valuevec(self, batch_rows):
        """dense (len(batch_rows) x ni) pop-weighted member-bag value vectors (for encoder-faithful init)."""
        return np.asarray(self.Mw[batch_rows].todense(), np.float32)


# ======================================================================== model
class UnifiedEncoder(nn.Module):
    """ONE encoder over a UNIFIED input: items AND concepts are the same kind of signed-value input dim.
    Input = concat[ item_val(ni), item_mask(ni), concept_val(C), concept_mask(C) ] -> 5 dense blocks -> z.
    A concept is ENCODED exactly like an item (no separate additive path). Warm-startable from a0c: the
    item columns copy a0c's fc1; the concept columns init 0 (so item-only == a0c exactly at init)."""
    def __init__(self, ni, C, hidden=600, latent=512):
        super().__init__()
        self.ni = ni; self.C = C
        din = 2 * (ni + C)
        self.fc1 = nn.Linear(din, hidden); self.ln1 = nn.LayerNorm(hidden, eps=0.1)
        self.fc2 = nn.Linear(hidden, hidden); self.ln2 = nn.LayerNorm(hidden, eps=0.1)
        self.fc3 = nn.Linear(hidden, hidden); self.ln3 = nn.LayerNorm(hidden, eps=0.1)
        self.fc4 = nn.Linear(hidden, hidden); self.ln4 = nn.LayerNorm(hidden, eps=0.1)
        self.fc5 = nn.Linear(hidden, hidden); self.ln5 = nn.LayerNorm(hidden, eps=0.1)
        self.fc_z = nn.Linear(hidden, latent)

    def forward(self, iv, cv, dropout=0.0):
        im = (iv != 0).to(iv.dtype); cm = (cv != 0).to(cv.dtype)
        x = torch.cat([iv, im, cv, cm], dim=-1)
        norm = x.pow(2).sum(-1, keepdim=True).sqrt().clamp_min(1e-8)
        x = x / norm
        x = F.dropout(x, p=dropout, training=self.training)
        h1 = self.ln1(swish(self.fc1(x)))
        h2 = self.ln2(swish(self.fc2(h1) + h1))
        h3 = self.ln3(swish(self.fc3(h2) + h1 + h2))
        h4 = self.ln4(swish(self.fc4(h3) + h1 + h2 + h3))
        h5 = self.ln5(swish(self.fc5(h4) + h1 + h2 + h3 + h4))
        return self.fc_z(h5)


class UnifiedAE(nn.Module):
    """Unified item+concept encoder + item-scoring decoder. forward(xv, cg, cm=None): xv=item values,
    cg=concept values (mask derived internally; cm ignored, kept for eval-fn signature compatibility)."""
    def __init__(self, ni, C, hidden=600, latent=512):
        super().__init__()
        self.ni = ni; self.C = C
        self.encoder = UnifiedEncoder(ni, C, hidden, latent)
        self.decoder = nn.Linear(latent, ni)

    def forward(self, xv, cg=None, cm=None):
        cv = cg if cg is not None else torch.zeros((xv.shape[0], self.C), dtype=xv.dtype)
        z = self.encoder(xv, cv, 0.0)
        return self.decoder(z), z

    @torch.no_grad()
    def warm_start_a03b(self, path=A03B):
        """Copy a0c SignedAE encoder+decoder; concept input columns init 0 (item-only == a0c at init)."""
        blob = torch.load(path, map_location="cpu"); sd = blob["model"]
        ni = self.ni
        # fc1: a0c input = [xv(ni), mask(ni)] = my [item_val, item_mask] = cols [0:2ni]; concept cols -> 0
        w = self.encoder.fc1.weight; w.zero_()
        w[:, :2 * ni] = sd["encoder.fc1.weight"]
        self.encoder.fc1.bias.copy_(sd["encoder.fc1.bias"])
        for k in ("fc2", "fc3", "fc4", "fc5", "fc_z", "ln1", "ln2", "ln3", "ln4", "ln5"):
            for suf in ("weight", "bias"):
                key = f"encoder.{k}.{suf}"
                if key in sd:
                    getattr(self.encoder, k).__getattr__(suf).copy_(sd[key])
        self.decoder.weight.copy_(sd["decoder.weight"]); self.decoder.bias.copy_(sd["decoder.bias"])
        log(f"[unified] warm-started a0c (val_full={blob.get('val_full')}); concept columns init 0 "
            f"(item-only path == a0c exactly)")


class ReconciledAE(nn.Module):
    """Item tower (SignedEncoder, warm-start a03b) + DEDICATED concept channel:
       z = z_item(xv) + (1/sqrt(n_c)) * W_shared( sum_c g_c e_c );  scores = decoder(z).
       e_c LEARNABLE (encoder-faithful member-bag init); W_shared LEARNABLE (identity init)."""
    def __init__(self, ni, C, hidden=600, latent=512):
        super().__init__()
        self.ni = ni; self.C = C; self.latent = latent
        self.encoder = SignedEncoder(ni, hidden, latent, use_mask=True)
        self.decoder = nn.Linear(latent, ni)
        self.e_c = nn.Parameter(torch.zeros(C, latent))            # init in init_from_a03b
        self.W_shared = nn.Linear(latent, latent, bias=False)
        with torch.no_grad():
            self.W_shared.weight.copy_(torch.eye(latent))          # identity init

    def item_z(self, xv):
        return self.encoder(xv, 0.0)

    def forward(self, xv, cg, cmask):
        """xv:(B,ni) signed item values. cg:(B,C) concept g-values (0 unanswered). cmask:(B,C) answered flag."""
        z = self.encoder(xv, 0.0)
        if cg is not None:
            pooled = cg @ self.e_c                                 # (B, latent) = sum_c g_c e_c
            # MEAN-pool (1/n_c), NOT 1/sqrt: e_c are LARGE and ALIGNED (taste deltas) so sqrt lets aligned
            # sums grow and swamp z_item at large n_c. Mean bounds total to ~one concept's magnitude for any
            # n_c; at n_c=1 (cold-start) it reproduces a single concept EXACTLY. (deviates from Fable A5 sqrt
            # for the alignment reason -- verified: sqrt gave full-profile catastrophe 0.42->0.0045.)
            nc = cmask.sum(-1, keepdim=True).clamp_min(1.0)
            z = z + self.W_shared(pooled) / nc
        return self.decoder(z), z

    @torch.no_grad()
    def init_from_a03b(self, cb, path=A03B, batch=256):
        """Warm-start encoder+decoder from a03b (signed_latent SignedAE). Then e_c = ENCODER-FAITHFUL
        member-bag delta: e_c[c] = encoder(memberbag_valuevec_c) - encoder(0). Reproduces the proven
        inference-injection response at step 0; then learned."""
        blob = torch.load(path, map_location="cpu")
        sd = blob["model"]
        enc_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
        dec_sd = {k[len("decoder."):]: v for k, v in sd.items() if k.startswith("decoder.")}
        self.encoder.load_state_dict(enc_sd); self.decoder.load_state_dict(dec_sd)
        self.eval()
        z0 = self.encoder(torch.zeros((1, self.ni), dtype=torch.float32), 0.0)[0]   # prior
        for b in range(0, cb.C, batch):
            rows = np.arange(b, min(b + batch, cb.C))
            mb = torch.from_numpy(cb.memberbag_valuevec(rows))     # (nb, ni) pop-weighted +weights
            zc = self.encoder(mb, 0.0)                             # (nb, latent)
            self.e_c[rows] = zc - z0[None, :]
        # rows with no members: encoder(0)->z0 so e_c = 0 (neutral); leave as zeros
        log(f"[init] warm-started a03b (val_full={blob.get('val_full')}); e_c encoder-faithful init "
            f"|e_c|mean={self.e_c.norm(dim=1).mean().item():.3f} (zeros for {int((~cb.has_members).sum())} memberless)")


# ======================================================================== A4 concept derivation (known-only)
def derive_concepts(inp_items, inp_stars, cb):
    """A4 (leak-safe, FAST): from the REVEALED/INPUT items+stars ONLY (target excluded by construction),
    derive concept answers. For each concept with >=1 input member: v_c = snapped mean-member star;
    k_c from #input members. Column-slice Mcsc[:, inp_items] = O(k * concepts-per-item), not O(nnz)."""
    inp_items = np.asarray(inp_items, np.int64)
    if len(inp_items) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.float32)
    sub = cb.Mcsc[:, inp_items]                                    # (C x k) sparse
    nknown = np.asarray(sub.sum(1)).ravel()
    ssum = np.asarray(sub.dot(np.asarray(inp_stars, np.float64))).ravel()
    active = np.where(nknown >= 1)[0]
    if len(active) == 0:
        return active, np.zeros(0, np.float32)
    mean_star = ssum[active] / nknown[active]
    vlev = np.array([star_to_val(s) for s in mean_star], np.int64)
    klev = np.where(nknown[active] >= 3, 2, 1).astype(np.int64)   # coverage -> know_well / rough (approx A4)
    g = (VALENCE[vlev] * CONF[klev]).astype(np.float32)
    return active, g


# ======================================================================== eval (strength + k-curve)
def eval_strength(model, cb, base, SPL, users, revealed="full", k=None, use_concepts=True, seedoff=0,
                  concept_sign=None, tail=False):
    """Full-catalog held-liked NDCG@10. Items = revealed profile-half likes (signed). Concepts = A4-derived
    from the REVEALED items. concept_sign: None=all concepts, 'pos'=only g>0 (loved/liked), 'neg'=only g<0
    (hated/meh) -> isolates positive-half vs negative-half concept contribution (Fable A1). Returns
    (full_ndcg, tail_ndcg, n)."""
    model.eval(); ni = base["ni"]; headmask = base["headmask"]; C = cb.C
    ff, tt = [], []
    rng = np.random.default_rng(4242 + seedoff)
    Xv, Cg, Cm, metas = [], [], [], []
    for u in users:
        profset, held, prof_r, held_r = SPL[u]
        tlike = [j for j in held if held_r[j] >= LO]
        if not tlike:
            continue
        liked_prof = [j for j in prof_r if prof_r[j] >= LO]
        if revealed == "full":
            idx = list(liked_prof)
        else:
            if len(liked_prof) < k:
                continue
            sel = rng.choice(len(liked_prof), size=k, replace=False)
            idx = [liked_prof[i] for i in sel]
        xv = np.zeros(ni, np.float32)
        for j in idx:
            xv[j] = scale_rating(prof_r[j])
        cg = np.zeros(C, np.float32); cm = np.zeros(C, np.float32)
        if use_concepts and idx:
            cidx, g = derive_concepts(np.array(idx, np.int64),
                                      np.array([prof_r[j] for j in idx], np.float64), cb)
            if concept_sign == "pos":
                keep = g > 0; cidx, g = cidx[keep], g[keep]
            elif concept_sign == "neg":
                keep = g < 0; cidx, g = cidx[keep], g[keep]
            if len(cidx):
                cg[cidx] = g; cm[cidx] = 1.0
        Xv.append(xv); Cg.append(cg); Cm.append(cm); metas.append((profset, tlike))
    if not Xv:
        return float("nan"), float("nan"), 0
    with torch.no_grad():
        scores = []
        Bsz = 256
        for b in range(0, len(Xv), Bsz):
            xv = torch.from_numpy(np.stack(Xv[b:b + Bsz]))
            cg = torch.from_numpy(np.stack(Cg[b:b + Bsz])) if use_concepts else None
            cm = torch.from_numpy(np.stack(Cm[b:b + Bsz])) if use_concepts else None
            sc = model(xv, cg, cm)[0].numpy().astype(np.float64)
            scores.append(sc)
        scores = np.concatenate(scores, 0)
    for r, (profset, tlike) in enumerate(metas):
        nf = ndcg10(scores[r], tlike, profset, headmask, False)
        nt = ndcg10(scores[r], tlike, profset, headmask, True)
        if nf is not None: ff.append(nf)
        if nt is not None: tt.append(nt)
    return (float(np.mean(ff)) if ff else float("nan"),
            float(np.mean(tt)) if tt else float("nan"), len(ff))


# ======================================================================== Phase-2 curriculum + training
# pre-registered curriculum constants (Fable pass-2 A3: item-masked fraction is a KNOB, not incidental)
W_FULL = 0.30        # fraction of examples that are full-profile item-only (strength retention)
P_ANS = 1.0          # Phase 2 is CLEAN: no refusals
P_KEEP_CONCEPT = 0.9 # prob an interview reveals its derivable concepts (else item-only short interview)
P_MASK = 0.40        # prob of the ITEM-MASKED concept-forcing slice (drop revealed concepts' member items)
TMAX = 20            # interview length cap (1..20, log-uniform)


def build_train_users_stars(base):
    """Per train user: item ids (sorted) + RAW stars + signed values + liked. Full data, no cap."""
    tra_u = base["tra_u"]; tra_i = base["tra_i"]; tra_r = base["tra_r"]
    order = np.argsort(tra_u, kind="stable")
    tra_u = tra_u[order]; tra_i = tra_i[order]; tra_r = tra_r[order]
    users = []
    N = len(tra_u); b = 0
    while b < N:
        e = b
        while e < N and tra_u[e] == tra_u[b]:
            e += 1
        its = tra_i[b:e].astype(np.int64); rs = tra_r[b:e].astype(np.float64)
        srt = np.argsort(its); its = its[srt]; rs = rs[srt]
        liked = its[rs >= LO]
        if len(liked) >= 2:
            users.append(dict(items=its, stars=rs.astype(np.float32),
                              sv=scale_rating(rs).astype(np.float32), liked=liked,
                              disliked=its[rs <= HI]))
        b = e
    return users


def make_p2_example(u, rng, cb, info):
    """ONE leak-safe Phase-2 example. Returns (item_idx, item_sval, cidx, cg, target) or None.
    - full-profile item-only (strength) with prob W_FULL (no concepts).
    - else CLEAN emulated interview: k items by strategy (real signed vals) + concepts DERIVED FROM THOSE
      input items (leak-free: target = liked \\ input), varying item:concept proportions, with a P_MASK
      ITEM-MASKED concept-forcing slice (drop revealed concepts' member items from the item channel; the
      dropped items are still 'known' so stay OUT of target). NO u['liked'] fallback (Fable A5): skip if empty."""
    its = u["items"]; n = len(its); liked = u["liked"]
    if rng.random() < W_FULL:
        drop = rng.uniform(0.0, 0.5)
        inp = its[rng.random(n) >= drop]
        target = np.setdiff1d(liked, inp, assume_unique=False)
        if len(target) == 0 or len(inp) == 0:
            return None
        sval = u["sv"][np.searchsorted(its, inp)]
        return inp, sval, np.zeros(0, np.int64), np.zeros(0, np.float32), target
    # interview
    k = int(round(math.exp(rng.uniform(0.0, math.log(max(2, min(n, TMAX)))))))
    k = max(1, min(k, n))
    strat = STRATEGIES[np.searchsorted(SL._CUM_STRAT, rng.random())]
    sel = np.asarray(select_items(u, k, strat, rng, info)).astype(np.int64)
    inp = np.unique(sel)
    if len(inp) == 0:
        return None
    inp_stars = u["stars"][np.searchsorted(its, inp)]
    inp_sval = u["sv"][np.searchsorted(its, inp)]
    # derive concepts from INPUT items (leak-free); reveal them with prob P_KEEP_CONCEPT
    cidx = np.zeros(0, np.int64); cg = np.zeros(0, np.float32)
    item_channel = inp; item_sval = inp_sval
    if rng.random() < P_KEEP_CONCEPT:
        cidx, cg = derive_concepts(inp, inp_stars, cb)
        if len(cidx):
            # keep a random proportion of the derivable concepts (varying item:concept mix)
            keep = rng.random(len(cidx)) < rng.uniform(0.3, 1.0)
            cidx = cidx[keep]; cg = cg[keep]
            # ITEM-MASKED concept-forcing slice: drop revealed concepts' member items from the item channel
            if len(cidx) and rng.random() < P_MASK:
                masked = set()
                for c in cidx:
                    masked.update(int(j) for j in cb.Mbin[int(c)].indices)
                keepmask = np.array([int(j) not in masked for j in inp], bool)
                item_channel = inp[keepmask]; item_sval = inp_sval[keepmask]
    # target = liked NOT in the ORIGINAL input (masked items are known -> excluded); no fallback
    target = np.setdiff1d(liked, inp, assume_unique=False)
    if len(target) == 0:
        return None
    return item_channel, item_sval, cidx, cg, target


RICH = "C:/dev/phd/casper/.cache/rich_signal"
SPLIT_SEED = 123                                                 # arena_core.SEED (build_cohorts convention)


def recs_for_uids(uids, seed=SPLIT_SEED):
    """Reconstruct the EXACT per-user known/held split the answerer tables were built with (build_cohorts):
    per-user rng(seed*1_000_003+uid), META-order items, first half known / second half held-liked (r>=LO).
    Aligned to the mm-table rows BY CONSTRUCTION (uid-driven). Returns list of (known dict, held-liked list)."""
    d = np.load(SL.META)
    uu = d["uu"].astype(np.int64); ii = d["ii"].astype(np.int64); rr = d["rr"].astype(np.float32)
    order = np.argsort(uu, kind="stable"); uu, ii, rr = uu[order], ii[order], rr[order]
    bnd = np.searchsorted(uu, np.arange(uu[-1] + 2))
    recs = []
    for u in uids:
        u = int(u); s, e = bnd[u], bnd[u + 1]
        its = ii[s:e]; rat = rr[s:e]
        ru = np.random.default_rng(seed * 1_000_003 + u)
        perm = ru.permutation(len(its)); half = len(its) // 2
        known = {int(its[k]): float(rat[k]) for k in perm[:half]}
        held = [int(its[k]) for k in perm[half:] if rat[k] >= LO]
        recs.append((known, held))
    return recs


def ans_concepts(know_row, val_row):
    """Answerer concept answers for one user: answered concept indices + graded g (valence*confidence).
    Answered = know>=1 AND val>=0 (refusals know=0 / val=-1 excluded). g = VALENCE[val]*CONF[know]."""
    ans = np.where((know_row >= 1) & (val_row >= 0))[0]
    if len(ans) == 0:
        return ans, np.zeros(0, np.float32)
    g = (VALENCE[val_row[ans].astype(np.int64)] * CONF[know_row[ans].astype(np.int64)]).astype(np.float32)
    return ans, g


def make_example_ans(rec, know_row, val_row, rng, w_item=0.4, w_conly=0.3):
    """ONE Phase-2 example from the ANSWERER's answers. item channel = user's KNOWN items (real signed value);
    concept channel = the answerer's answered concepts (all of them = 'all answers'); target = held-liked.
    Mixed regime: item-only (strength) / concept-only (forcing) / mixed."""
    known, held = rec
    if not held:
        return None
    target = np.asarray(held, np.int64)
    ki = np.fromiter(known.keys(), np.int64, len(known))
    ksv = scale_rating(np.fromiter(known.values(), np.float64, len(known))).astype(np.float32)
    ans, g = ans_concepts(know_row, val_row)
    r = rng.random()
    if r < w_item or len(ans) == 0:                              # item-only (strength)
        return ki, ksv, np.zeros(0, np.int64), np.zeros(0, np.float32), target
    if r < w_item + w_conly:                                     # concept-only (forcing)
        return np.zeros(0, np.int64), np.zeros(0, np.float32), ans, g, target
    return ki, ksv, ans, g, target                              # mixed


def eval_concept_only_ans(model, cb, recs, know_mm, val_mm, kc, ni, headmask):
    """Concept-only cold-start from ANSWERER answers: reveal top-kc answered concepts (know_well first),
    NO items -> NDCG@10 held-liked, full+tail. Leak-safe (concepts from known-half; target held-half)."""
    model.eval(); ff, tt = [], []
    Xv, Cg, metas = [], [], []
    for i, (known, held) in enumerate(recs):
        if not held:
            continue
        profset = set(known.keys())
        ans, g = ans_concepts(know_mm[i, :cb.C], val_mm[i, :cb.C])
        if len(ans) == 0:
            continue
        order = np.argsort(-know_mm[i, :cb.C][ans].astype(np.int64))[:kc]   # know_well first
        sel = ans[order]; gsel = g[order]
        cg = np.zeros(cb.C, np.float32); cg[sel] = gsel
        Xv.append(np.zeros(ni, np.float32)); Cg.append(cg); metas.append((profset, held))
    if not Xv:
        return float("nan"), float("nan")
    with torch.no_grad():
        sc = []
        for b in range(0, len(Xv), 256):
            s = model(torch.from_numpy(np.stack(Xv[b:b + 256])),
                      torch.from_numpy(np.stack(Cg[b:b + 256])))[0].numpy().astype(np.float64)
            sc.append(s)
        sc = np.concatenate(sc, 0)
    for r, (profset, held) in enumerate(metas):
        nf = ndcg10(sc[r], held, profset, headmask, False)
        nt = ndcg10(sc[r], held, profset, headmask, True)
        if nf is not None: ff.append(nf)
        if nt is not None: tt.append(nt)
    return (float(np.mean(ff)) if ff else float("nan"), float(np.mean(tt)) if tt else float("nan"))


def cmd_p2ans(args):
    """PHASE 2 on the ANSWERER'S REAL answers (mm_train tables), concepts encoded like items (UnifiedAE).
    Confidence = the answerer's knowledge level (learnable). NOT the oracle member-mean. Warm-start a0c."""
    base = load_arena_base(); ni = base["ni"]; headmask = base["headmask"]
    SL._POP = base["cnt"].astype(np.float64)
    cb = ConceptBank(ni, base["cnt"])
    know = np.load(f"{RICH}/mm_train_know.npy", mmap_mode="r"); val = np.load(f"{RICH}/mm_train_val.npy", mmap_mode="r")
    uids = np.load(f"{RICH}/mm_train_uids.npy")
    vk = np.load(f"{RICH}/mm_val_know.npy", mmap_mode="r"); vv = np.load(f"{RICH}/mm_val_val.npy", mmap_mode="r")
    vuids = np.load(f"{RICH}/mm_val_uids.npy")
    assert know.shape[1] >= cb.C and know.shape[0] == len(uids), "table/uid shape mismatch"
    log(f"[p2ans] answerer tables: train {know.shape} val {vk.shape}; reconstructing split targets ...")
    tr = recs_for_uids(uids); va = recs_for_uids(vuids)
    ntr_ok = sum(1 for _, h in tr if h); nva_ok = sum(1 for _, h in va if h)
    log(f"[p2ans] targets: {ntr_ok}/{len(tr)} train users with held-liked, {nva_ok}/{len(va)} val")
    model = UnifiedAE(ni, cb.C); model.warm_start_a03b()
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    tag = args.tag; ckpt = os.path.join(OUTDIR, f"{tag}.pt"); best_ckpt = os.path.join(OUTDIR, f"{tag}_best.pt")
    # baselines
    ic_f, ic_t = eval_concept_only_ans(model, cb, va, vk, vv, 0, ni, headmask)  # kc=0 -> intercept
    f0 = eval_strength(model, cb, base, build_splits(base, SEEDS[0]), cohort(base, build_splits(base, SEEDS[0]), "val"), "full", use_concepts=False)[0]
    log(f"[p2ans] INIT intercept full={ic_f:.4f} tail={ic_t:.4f}  full-profile(item-only)={f0:.4f}")
    kk = (1, 2, 4, 8, 16, 32); B = 256; idxlist = np.arange(len(tr)); best = -1.0
    for ep in range(args.epochs):
        model.train(); rng = np.random.default_rng(ep * 7919 + 5)
        np.random.default_rng(ep).shuffle(idxlist)
        t0 = time.time(); running = 0.0; nb = 0
        for st in range(0, len(tr), B):
            bat = idxlist[st:st + B]
            iv = torch.zeros((len(bat), ni), dtype=torch.float32)
            cv = torch.zeros((len(bat), cb.C), dtype=torch.float32)
            tgt = torch.zeros((len(bat), ni), dtype=torch.float32); npos = 0
            for r, i in enumerate(bat):
                ex = make_example_ans(tr[i], know[i, :cb.C], val[i, :cb.C], rng)
                if ex is None:
                    continue
                it_idx, it_val, cidx, gg, target = ex
                if len(it_idx): iv[r, it_idx] = torch.from_numpy(it_val)
                if len(cidx): cv[r, cidx] = torch.from_numpy(gg)
                tgt[r, target] = 1.0; npos += 1
            if npos == 0:
                continue
            logits, z = model(iv, cv)
            logsm = F.log_softmax(logits, dim=-1)
            loss = -((logsm * tgt).sum(-1) / tgt.sum(-1).clamp_min(1.0)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            running += float(loss); nb += 1
            if nb % 150 == 0:
                log(f"  [p2ans] ep{ep} b{nb}/{(len(tr)+B-1)//B} loss={running/nb:.4f} {(time.time()-t0)/60:.1f}m")
        conly = {k: eval_concept_only_ans(model, cb, va, vk, vv, k, ni, headmask) for k in kk}
        full = eval_strength(model, cb, base, build_splits(base, SEEDS[0]), cohort(base, build_splits(base, SEEDS[0]), "val"), "full", use_concepts=False)[0]
        log(f"[p2ans ep{ep+1}] loss={running/max(nb,1):.4f} FULL(item)={full:.4f} | concept-only " +
            " ".join(f"k{k}={conly[k][0]:.4f}/{conly[k][1]:.4f}" for k in kk) +
            f"  (int {ic_f:.4f}/{ic_t:.4f}) ({(time.time()-t0)/60:.1f}m)")
        c8f, c8t = conly[8]
        log(f"[p2ans] k8 concept-only full={c8f:.4f} ({c8f-ic_f:+.4f}) TAIL={c8t:.4f} ({c8t-ic_t:+.4f})  "
            f"strength {'OK' if full>=0.486 else 'DROP'} ({full:.4f})")
        safe_save({"model": model.state_dict(), "epoch": ep + 1, "conly": conly, "full": full}, ckpt)
        sc = c8t + (full if full >= 0.486 else 0.0)
        if sc > best:
            best = sc; safe_save({"model": model.state_dict(), "epoch": ep + 1, "conly": conly, "full": full}, best_ckpt)
            open(os.path.join(OUTDIR, f"{tag}_peak.txt"), "w").write(
                f"PEAK k8 tail={c8t:.4f} (int {ic_t:.4f}) full={full:.4f} @ep{ep+1}\n")
    log(f"[p2ans] done tag={tag}")


def make_unified_example(u, rng, cb, info, w_item=0.4, w_conly=0.3):
    """ONE example for the unified encoder. Reveal k items by strategy; derive concepts from them.
    Regime: item-only (strength) / concept-only (forcing) / mixed. target = liked NOT revealed (leak-safe).
    Returns (item_idx, item_sval, cidx, cg, target) with empty arrays for the absent channel."""
    its = u["items"]; n = len(its); liked = u["liked"]
    k = int(round(math.exp(rng.uniform(0.0, math.log(max(2, min(n, TMAX)))))))
    k = max(1, min(k, n))
    strat = STRATEGIES[np.searchsorted(SL._CUM_STRAT, rng.random())]
    sel = np.unique(np.asarray(select_items(u, k, strat, rng, info)).astype(np.int64))
    if len(sel) == 0:
        return None
    target = np.setdiff1d(liked, sel, assume_unique=False)
    if len(target) == 0:
        return None
    sel_sval = u["sv"][np.searchsorted(its, sel)]
    sel_stars = u["stars"][np.searchsorted(its, sel)]
    cidx, g = derive_concepts(sel, sel_stars, cb)
    r = rng.random()
    z_i = np.zeros(0, np.int64); z_f = np.zeros(0, np.float32)
    if r < w_item:                                       # item-only (strength)
        return sel, sel_sval, z_i, z_f, target
    if r < w_item + w_conly:                             # concept-only (forcing)
        if len(cidx) == 0:
            return None
        return z_i, z_f, cidx, g, target
    if len(cidx) == 0:                                   # mixed (fall back to item-only if no concepts)
        return sel, sel_sval, z_i, z_f, target
    return sel, sel_sval, cidx, g, target


def cmd_utrain(args):
    """UNIFIED encoder: concepts treated EXACTLY like items (extra input dims, same encoder). Warm-start a0c
    (item-only == a0c at init), CO-TRAIN encoder+decoder on a mixed curriculum (item-only / concept-only /
    mixed). Track full-profile strength AND concept-only-vs-intercept every epoch."""
    base = load_arena_base(); ni = base["ni"]
    SL._POP = base["cnt"].astype(np.float64); info = item_info(base)
    cb = ConceptBank(ni, base["cnt"])
    users = build_train_users_stars(base)
    model = UnifiedAE(ni, cb.C); model.warm_start_a03b()
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    tag = args.tag
    ckpt = os.path.join(OUTDIR, f"{tag}.pt"); best_ckpt = os.path.join(OUTDIR, f"{tag}_best.pt")
    SPLv = build_splits(base, SEEDS[0]); vusers = cohort(base, SPLv, "val")
    ic = eval_intercept(model, cb, base, SPLv, vusers)[0]
    f0 = eval_strength(model, cb, base, SPLv, vusers, "full", use_concepts=False)[0]
    log(f"[utrain] INIT intercept={ic:.4f}  full-profile(item-only)={f0:.4f}  (a0c 0.4961)")
    kk = (1, 2, 4, 8, 16, 32)
    B = 256; idxlist = np.arange(len(users)); best = -1.0
    for ep in range(args.epochs):
        model.train(); rng = np.random.default_rng(ep * 7919 + 91)
        np.random.default_rng(ep).shuffle(idxlist)
        t0 = time.time(); running = 0.0; nb = 0; nskip = 0
        for st in range(0, len(users), B):
            bat = idxlist[st:st + B]
            iv = torch.zeros((len(bat), ni), dtype=torch.float32)
            cv = torch.zeros((len(bat), cb.C), dtype=torch.float32)
            tgt = torch.zeros((len(bat), ni), dtype=torch.float32); npos = 0
            for r, ui in enumerate(bat):
                ex = make_unified_example(users[ui], rng, cb, info)
                if ex is None:
                    nskip += 1; continue
                it_idx, it_val, cidx, gg, target = ex
                if len(it_idx):
                    iv[r, it_idx] = torch.from_numpy(it_val)
                if len(cidx):
                    cv[r, cidx] = torch.from_numpy(gg)
                tgt[r, target] = 1.0; npos += 1
            if npos == 0:
                continue
            logits, z = model(iv, cv)
            logsm = F.log_softmax(logits, dim=-1)
            denom = tgt.sum(-1).clamp_min(1.0)
            loss = -((logsm * tgt).sum(-1) / denom).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            running += float(loss); nb += 1
            if nb % 150 == 0:
                log(f"  [utrain] ep{ep} b{nb}/{(len(users)+B-1)//B} loss={running/nb:.4f} {(time.time()-t0)/60:.1f}m")
        conly = {kc: eval_concept_only(model, cb, base, SPLv, vusers, kc)[0] for kc in kk}
        full = eval_strength(model, cb, base, SPLv, vusers, "full", use_concepts=False)[0]
        log(f"[utrain ep{ep+1}] loss={running/max(nb,1):.4f} FULL(item)={full:.4f} | concept-only " +
            " ".join(f"k{kc}={conly[kc]:.4f}" for kc in kk) + f" (intercept {ic:.4f}) ({(time.time()-t0)/60:.1f}m)")
        c8 = conly[8]
        log(f"[utrain] concept-only {'*** ABOVE INTERCEPT' if c8 > ic else 'below intercept'} "
            f"(k8={c8:.4f} vs {ic:.4f}, {c8-ic:+.4f})  strength {'OK' if full>=0.486 else 'DROPPED'} ({full:.4f})")
        safe_save({"model": model.state_dict(), "epoch": ep + 1, "concept_only": conly, "full": full,
                   "intercept": ic}, ckpt)
        score = c8 + (full if full >= 0.486 else 0.0)             # joint: concept-only lift + strength held
        if score > best:
            best = score
            safe_save({"model": model.state_dict(), "epoch": ep + 1, "concept_only": conly, "full": full,
                       "intercept": ic}, best_ckpt)
            open(os.path.join(OUTDIR, f"{tag}_peak.txt"), "w").write(
                f"PEAK concept-only k8={c8:.4f} (int {ic:.4f}, {c8-ic:+.4f}) full={full:.4f} @ep{ep+1}\n")
    log(f"[utrain] done tag={tag}")


def make_concept_only_example(u, rng, cb, info):
    """Reveal k items by strategy -> derive concepts from them -> feed ONLY concepts (NO item channel).
    target = liked NOT in the revealed set (leak-safe). Forces the concept path to carry the belief."""
    its = u["items"]; n = len(its); liked = u["liked"]
    k = int(round(math.exp(rng.uniform(0.0, math.log(max(2, min(n, TMAX)))))))
    k = max(1, min(k, n))
    strat = STRATEGIES[np.searchsorted(SL._CUM_STRAT, rng.random())]
    sel = np.unique(np.asarray(select_items(u, k, strat, rng, info)).astype(np.int64))
    if len(sel) == 0:
        return None
    sel_stars = u["stars"][np.searchsorted(its, sel)]
    cidx, g = derive_concepts(sel, sel_stars, cb)
    if len(cidx) == 0:
        return None
    target = np.setdiff1d(liked, sel, assume_unique=False)
    if len(target) == 0:
        return None
    return cidx, g, target


def cmd_ridge(args):
    """CHEAPEST DECISIVE TEST (closed-form, NO training loop). Whitened concept->item map on FROZEN a0c
    features. phi(i)=a0c decoder factor; v_c=pop-weighted member centroid in phi-space; q_u=sum_c a_c v_c.
    Fit W (ridge) so Phi@(W q_u) ranks held-liked. Compare W (whitened) vs W=I (raw align/PEBOL-static) vs
    intercept, on cold-start k-curve, FULL and TAIL. Question: does whitened concept score beat intercept ON
    THE TAIL? If not -> concept channel refuted before any training."""
    base = load_arena_base(); ni = base["ni"]; headmask = base["headmask"]
    SL._POP = base["cnt"].astype(np.float64)
    cb = ConceptBank(ni, base["cnt"])
    blob = torch.load(A03B, map_location="cpu")
    Phi = blob["model"]["decoder.weight"].numpy().astype(np.float64)          # (ni, d)
    bias = blob["model"]["decoder.bias"].numpy().astype(np.float64)
    d = Phi.shape[1]
    Vc = np.asarray(cb.Mw @ Phi)                                              # (C, d) concept = member centroid
    beta = 200.0
    Ainv = np.linalg.inv(Phi.T @ Phi + beta * np.eye(d))
    Wpath = os.path.join(OUTDIR, "ridge_W.npy")
    if os.path.exists(Wpath):
        W = np.load(Wpath); log(f"[ridge] loaded cached W |W|={np.linalg.norm(W):.2f}")
    else:
        users = build_train_users_stars(base)
        log(f"[ridge] fitting W over {len(users)} train users (d={d}, beta={beta}) -- NO cap")
        Q = np.zeros((len(users), d)); Z = np.zeros((len(users), d)); m = 0
        t0 = time.time()
        for u in users:
            cidx, g = derive_concepts(u["items"], u["stars"], cb)             # concepts from full profile
            if len(cidx) == 0:
                continue
            Q[m] = g @ Vc[cidx]                                               # q_u = sum a_c v_c
            Z[m] = Ainv @ Phi[u["liked"]].sum(0)                             # z*_u = ideal item-space belief
            m += 1
            if m % 40000 == 0:
                log(f"  [ridge] accumulated {m} users {(time.time()-t0)/60:.1f}m")
        Q = Q[:m]; Z = Z[:m]
        W = (Z.T @ Q) @ np.linalg.inv(Q.T @ Q + beta * np.eye(d))            # ridge map q->z*
        np.save(Wpath, W)
        log(f"[ridge] W fit on {m} users |W|={np.linalg.norm(W):.2f} saved ({(time.time()-t0)/60:.1f}m)")

    def eval_map(M, K):
        SPL = build_splits(base, SEEDS[0]); tusers = cohort(base, SPL, "test")
        ff, tt = [], []
        for u in tusers:
            profset, held, prof_r, held_r = SPL[u]
            tlike = [j for j in held if held_r[j] >= LO]
            if not tlike:
                continue
            prof_items = np.array(list(prof_r.keys()), np.int64)
            prof_stars = np.array([prof_r[j] for j in prof_items], np.float64)
            cidx, g = derive_concepts(prof_items, prof_stars, cb)
            if len(cidx) == 0:
                continue
            sub = cb.Mcsc[:, prof_items]; nknown = np.asarray(sub.sum(1)).ravel()[cidx]
            sel = np.argsort(-nknown)[:K]
            q = g[sel] @ Vc[cidx[sel]]
            score = (Phi @ (M @ q)) + bias
            nf = ndcg10(score, tlike, profset, headmask, False)
            nt = ndcg10(score, tlike, profset, headmask, True)
            if nf is not None: ff.append(nf)
            if nt is not None: tt.append(nt)
        return (float(np.mean(ff)) if ff else float("nan"),
                float(np.mean(tt)) if tt else float("nan"))
    # intercept (mostpop-ish): a0c decoder bias ranking
    SPL = build_splits(base, SEEDS[0]); tusers = cohort(base, SPL, "test")
    ic_f, ic_t = [], []
    for u in tusers:
        profset, held, prof_r, held_r = SPL[u]
        tlike = [j for j in held if held_r[j] >= LO]
        if tlike:
            nf = ndcg10(bias.copy(), tlike, profset, headmask, False)
            nt = ndcg10(bias.copy(), tlike, profset, headmask, True)
            if nf is not None: ic_f.append(nf)
            if nt is not None: ic_t.append(nt)
    log(f"[ridge] INTERCEPT full={np.mean(ic_f):.4f} tail={np.mean(ic_t):.4f}")
    for K in (1, 2, 4, 8, 16):
        wf, wt = eval_map(W, K)
        af, at = eval_map(np.eye(d), K)
        log(f"[ridge] K={K:2d} | WHITENED full={wf:.4f} tail={wt:.4f} | W=I(align) full={af:.4f} tail={at:.4f}")
    log("[ridge] DECISIVE: WHITENED tail > intercept tail => concept channel has signal")


def cmd_cctrain(args):
    """AUTHOR RECIPE: freeze the good item model (encoder+decoder from a0c); train ONLY the concept path
    (e_c + W_shared) on CONCEPT-ONLY input, so concepts are FORCED to learn to raise the belief above the
    intercept when no items are known. Unified space: concepts live in the same z-space the decoder scores."""
    base = load_arena_base(); ni = base["ni"]
    SL._POP = base["cnt"].astype(np.float64); info = item_info(base)
    cb = ConceptBank(ni, base["cnt"])
    users = build_train_users_stars(base)
    model = ReconciledAE(ni, cb.C); model.init_from_a03b(cb)
    # FREEZE the item model; train ONLY the concept path
    for p in model.encoder.parameters(): p.requires_grad_(False)
    for p in model.decoder.parameters(): p.requires_grad_(False)
    model.e_c.requires_grad_(True)
    for p in model.W_shared.parameters(): p.requires_grad_(True)
    trainable = [model.e_c] + list(model.W_shared.parameters())
    nptr = sum(p.numel() for p in trainable)
    opt = torch.optim.Adam(trainable, lr=1e-3)                     # few params, frozen rest -> higher LR ok
    log(f"[cctrain] FROZEN item model; trainable concept params={nptr} (e_c {tuple(model.e_c.shape)} + W_shared)")
    tag = args.tag
    ckpt = os.path.join(OUTDIR, f"{tag}.pt"); best_ckpt = os.path.join(OUTDIR, f"{tag}_best.pt")
    SPLv = build_splits(base, SEEDS[0]); vusers = cohort(base, SPLv, "val")
    ic = eval_intercept(model, cb, base, SPLv, vusers)[0]
    log(f"[cctrain] INTERCEPT (no items, no concepts) NDCG@10={ic:.4f}  <-- the bar to beat with concepts")
    kk = (1, 2, 4, 8, 16, 32)
    B = 256; idxlist = np.arange(len(users)); best = -1.0
    for ep in range(args.epochs):
        model.train(); rng = np.random.default_rng(ep * 7919 + 13)
        np.random.default_rng(ep).shuffle(idxlist)
        t0 = time.time(); running = 0.0; nb = 0; nskip = 0
        for st in range(0, len(users), B):
            bat = idxlist[st:st + B]
            xv = torch.zeros((len(bat), ni), dtype=torch.float32)  # ALWAYS zero: concept-only
            cg = torch.zeros((len(bat), cb.C), dtype=torch.float32)
            cm = torch.zeros((len(bat), cb.C), dtype=torch.float32)
            tgt = torch.zeros((len(bat), ni), dtype=torch.float32); npos = 0
            for r, ui in enumerate(bat):
                ex = make_concept_only_example(users[ui], rng, cb, info)
                if ex is None:
                    nskip += 1; continue
                cidx, g, target = ex
                cg[r, cidx] = torch.from_numpy(g); cm[r, cidx] = 1.0
                tgt[r, target] = 1.0; npos += 1
            if npos == 0:
                continue
            logits, z = model(xv, cg, cm)
            logsm = F.log_softmax(logits, dim=-1)
            denom = tgt.sum(-1).clamp_min(1.0)
            loss = -((logsm * tgt).sum(-1) / denom).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            running += float(loss); nb += 1
            if nb % 150 == 0:
                log(f"  [cctrain] ep{ep} batch {nb}/{(len(users)+B-1)//B} loss={running/nb:.4f} {(time.time()-t0)/60:.1f}m")
        cur = {kc: eval_concept_only(model, cb, base, SPLv, vusers, kc)[0] for kc in kk}
        peakk = cur[8]
        log(f"[cctrain ep{ep+1}] loss={running/max(nb,1):.4f} concept-only " +
            " ".join(f"k{kc}={cur[kc]:.4f}" for kc in kk) + f"  (intercept {ic:.4f}) ({(time.time()-t0)/60:.1f}m)")
        safe_save({"model": model.state_dict(), "epoch": ep + 1, "concept_only_k8": peakk,
                   "intercept": ic, "curve": cur}, ckpt)
        if peakk > best:
            best = peakk
            safe_save({"model": model.state_dict(), "epoch": ep + 1, "concept_only_k8": peakk,
                       "intercept": ic, "curve": cur}, best_ckpt)
            open(os.path.join(OUTDIR, f"{tag}_peak.txt"), "w").write(
                f"PEAK concept-only k8={peakk:.4f} (intercept {ic:.4f}, +{peakk-ic:+.4f}) @ep{ep+1}\n")
        log(f"[cctrain] {'*** ABOVE INTERCEPT' if peakk>ic else 'still below intercept'} "
            f"(k8={peakk:.4f} vs {ic:.4f}, {peakk-ic:+.4f})")
    log(f"[cctrain] done best concept-only k8={best:.4f} vs intercept {ic:.4f}")


def val_metrics(model, cb, base, SPL, users):
    """Joint peak inputs: full-profile strength (item-only, guard) + cold-start early-AUC (items+concepts)."""
    kf = {}
    for k in (1, 2, 4, 8):
        kf[k] = eval_strength(model, cb, base, SPL, users, "k", k=k, use_concepts=True)[0]
    ks = [1, 2, 4, 8]; vals = [kf[k] for k in ks]
    auc = float(np.trapz(vals, ks) / (ks[-1] - ks[0]))
    full = eval_strength(model, cb, base, SPL, users, "full", use_concepts=False)[0]
    return full, auc, kf


def cmd_p2(args):
    base = load_arena_base(); ni = base["ni"]
    SL._POP = base["cnt"].astype(np.float64); info = item_info(base)
    cb = ConceptBank(ni, base["cnt"])
    users = build_train_users_stars(base)
    log(f"[p2] train users={len(users)} ni={ni} C={cb.C}  W_FULL={W_FULL} P_MASK={P_MASK} TMAX={TMAX}")
    model = ReconciledAE(ni, cb.C); model.init_from_a03b(cb)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)             # fine-tune from a0c: modest LR
    tag = args.tag
    ckpt = os.path.join(OUTDIR, f"{tag}.pt"); best_ckpt = os.path.join(OUTDIR, f"{tag}_best.pt")
    peak_txt = os.path.join(OUTDIR, f"{tag}_peak.txt")
    SPLv = build_splits(base, SEEDS[0]); vusers = cohort(base, SPLv, "val")
    start_ep = 0; best_auc = -1.0; history = []
    STRENGTH_FLOOR = 0.4961 - 0.010                                 # joint-peak guard (Fable A6)
    if args.resume and os.path.exists(ckpt):
        try:
            blob = torch.load(ckpt, map_location="cpu")
            model.load_state_dict(blob["model"]); opt.load_state_dict(blob.get("opt", opt.state_dict()))
            start_ep = blob.get("epoch", 0); history = blob.get("history", [])
            best_auc = max([h.get("val_auc", -1.0) for h in history], default=-1.0)
            log(f"[p2] RESUMED ep{start_ep} best_auc={best_auc:.4f}")
        except (RuntimeError, OSError, KeyError, EOFError) as e:
            log(f"[p2] resume failed ({e}); FRESH")
    # baseline val at init (t=0)
    if start_ep == 0:
        f0, a0, kf0 = val_metrics(model, cb, base, SPLv, vusers)
        log(f"[p2] INIT val full={f0:.4f} auc={a0:.4f} " + " ".join(f"k{k}={kf0[k]:.4f}" for k in (1,2,4,8)))
    B = 256
    idxlist = np.arange(len(users))
    for ep in range(start_ep, args.epochs):
        model.train(); rng = np.random.default_rng(ep * 7919 + 4242)
        np.random.default_rng(ep).shuffle(idxlist)
        t0 = time.time(); running = 0.0; nb = 0; nskip = 0
        for st in range(0, len(users), B):
            bat = idxlist[st:st + B]
            xv = torch.zeros((len(bat), ni), dtype=torch.float32)
            cg = torch.zeros((len(bat), cb.C), dtype=torch.float32)
            cm = torch.zeros((len(bat), cb.C), dtype=torch.float32)
            tgt = torch.zeros((len(bat), ni), dtype=torch.float32)
            npos = 0
            for r, ui in enumerate(bat):
                ex = make_p2_example(users[ui], rng, cb, info)
                if ex is None:
                    nskip += 1; continue
                it_idx, it_val, cidx, gg, target = ex
                if len(it_idx):
                    xv[r, it_idx] = torch.from_numpy(it_val)
                if len(cidx):
                    cg[r, cidx] = torch.from_numpy(gg); cm[r, cidx] = 1.0
                tgt[r, target] = 1.0; npos += 1
            if npos == 0:
                continue
            logits, z = model(xv, cg, cm)
            logsm = F.log_softmax(logits, dim=-1)
            denom = tgt.sum(-1).clamp_min(1.0)
            loss = -((logsm * tgt).sum(-1) / denom).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            running += float(loss); nb += 1
            if nb % 100 == 0:
                log(f"  [p2] ep{ep} batch {nb}/{(len(users)+B-1)//B} loss={running/nb:.4f} "
                    f"skip={nskip} {(time.time()-t0)/60:.1f}m")
        full, auc, kf = val_metrics(model, cb, base, SPLv, vusers)
        history.append({"epoch": ep + 1, "loss": running / max(nb, 1), "val_full": full,
                        "val_auc": auc, "kcurve": kf})
        log(f"[p2 ep{ep+1}] loss={running/max(nb,1):.4f} VAL full={full:.4f} auc={auc:.4f} "
            + " ".join(f"k{k}={kf[k]:.4f}" for k in (1,2,4,8)) + f" ({(time.time()-t0)/60:.1f}m)")
        safe_save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep + 1,
                   "history": history}, ckpt)
        # JOINT peak: best cold-start AUC subject to strength floor (Fable A6)
        if full >= STRENGTH_FLOOR and auc > best_auc:
            best_auc = auc
            safe_save({"model": model.state_dict(), "epoch": ep + 1, "val_full": full, "val_auc": auc,
                       "kcurve": kf}, best_ckpt)
            try:
                open(peak_txt, "w").write(f"PEAK auc={auc:.4f} full={full:.4f} @ep{ep+1} "
                                          f"(floor={STRENGTH_FLOOR:.4f})\n")
            except OSError:
                pass
            log(f"  [p2] *** JOINT PEAK auc={auc:.4f} (full={full:.4f} >= floor) saved")
        try:
            json.dump(history, open(os.path.join(OUTDIR, f"{tag}_history.json"), "w"))
        except OSError:
            pass
    log(f"[p2] done tag={tag} best_auc={best_auc:.4f}")


def _auc(curve, kk):
    return float(np.trapz([curve[k] for k in kk], kk) / (kk[-1] - kk[0]))


def gate_sweep(model, cb, base, n_probe=16):
    """Behavioral (SUPPORTING, not headline): sweep a concept hated->meh->liked->loved via the concept
    channel (empty items) -> its members' percentile should rise MONOTONICALLY; untouched concepts stay put
    (specificity). Trained-model version of the t=0 check."""
    model.eval(); ni = base["ni"]; C = cb.C
    sizes = np.asarray(cb.Mbin.sum(1)).ravel().copy(); sizes[cb.ntag:] = 0
    probe_c = np.argsort(-sizes)[:n_probe]
    z0 = np.zeros(ni, np.float32)
    levels = [(0, 2), (1, 2), (2, 2), (3, 2)]                 # (val_level, know_well) hated->loved
    rows = []; mono = 0; ok = 0
    ctrl_disp = []
    for c in probe_c:
        members = cb.Mbin[int(c)].indices
        if len(members) < 20:
            continue
        ctrl = int(probe_c[(list(probe_c).index(c) + 5) % len(probe_c)])   # a different concept = control
        ctrl_mem = cb.Mbin[ctrl].indices
        pcts = []; ctrl_pcts = []
        for vl, kl in levels:
            cg = np.zeros(C, np.float32); cm = np.zeros(C, np.float32); cm[c] = 1.0
            cg[c] = VALENCE[vl] * CONF[kl]
            with torch.no_grad():
                s = model(torch.from_numpy(z0[None]), torch.from_numpy(cg[None]),
                          torch.from_numpy(cm[None]))[0].numpy()[0].astype(np.float64)
            pcts.append(percentile_rank(s, members)); ctrl_pcts.append(percentile_rank(s, ctrl_mem))
        is_mono = all(pcts[i + 1] >= pcts[i] - 1e-6 for i in range(3))
        mono += int(is_mono); ok += 1
        ctrl_disp.append(max(ctrl_pcts) - min(ctrl_pcts))
        rows.append(dict(concept=cb.names[int(c)], pct_by_level=pcts, monotone=is_mono))
    return dict(monotone_frac=mono / max(ok, 1), n=ok, mean_ctrl_displacement=float(np.mean(ctrl_disp)) if ctrl_disp else float("nan"),
                per_concept=rows[:8])


def eval_concept_only(model, cb, base, SPL, users, kc, seedoff=0):
    """CONCEPT-ONLY cold-start: reveal kc CONCEPTS (graded, derived from the user's profile-half ratings),
    NO item channel -> NDCG@10 on held-liked. Isolates the concept channel's standalone taste-recovery.
    Concepts revealed in order of engagement (#known members desc = most-answerable first). Leak-free:
    concepts derived from profile-half only; target = held-liked."""
    model.eval(); ni = base["ni"]; headmask = base["headmask"]; Cn = cb.C
    ff, tt = [], []
    Xv, Cg, Cm, metas = [], [], [], []
    for u in users:
        profset, held, prof_r, held_r = SPL[u]
        tlike = [j for j in held if held_r[j] >= LO]
        if not tlike:
            continue
        prof_items = np.array(list(prof_r.keys()), np.int64)
        prof_stars = np.array([prof_r[j] for j in prof_items], np.float64)
        cidx, g = derive_concepts(prof_items, prof_stars, cb)
        if len(cidx) == 0:
            continue
        # order by engagement (#known members) desc
        sub = cb.Mcsc[:, prof_items]
        nknown = np.asarray(sub.sum(1)).ravel()[cidx]
        order = np.argsort(-nknown)
        sel = order[:kc]
        cg = np.zeros(Cn, np.float32); cm = np.zeros(Cn, np.float32)
        cg[cidx[sel]] = g[sel]; cm[cidx[sel]] = 1.0
        Xv.append(np.zeros(ni, np.float32)); Cg.append(cg); Cm.append(cm); metas.append((profset, tlike))
    if not Xv:
        return float("nan"), float("nan"), 0
    with torch.no_grad():
        scores = []
        for b in range(0, len(Xv), 256):
            sc = model(torch.from_numpy(np.stack(Xv[b:b + 256])),
                       torch.from_numpy(np.stack(Cg[b:b + 256])),
                       torch.from_numpy(np.stack(Cm[b:b + 256])))[0].numpy().astype(np.float64)
            scores.append(sc)
        scores = np.concatenate(scores, 0)
    for r, (profset, tlike) in enumerate(metas):
        nf = ndcg10(scores[r], tlike, profset, headmask, False)
        nt = ndcg10(scores[r], tlike, profset, headmask, True)
        if nf is not None: ff.append(nf)
        if nt is not None: tt.append(nt)
    return (float(np.mean(ff)) if ff else float("nan"),
            float(np.mean(tt)) if tt else float("nan"), len(ff))


def eval_bag_injection(model_old, cb, base, SPL, users, kc):
    """PREVIOUS-ITERATION baseline: reveal kc concepts, fold them into the a0c ITEM channel as a MEMBER-BAG
    (each concept's members set to its graded g), feed the OLD item-only SignedAE (NO concept channel).
    kc=0 -> empty -> the a0c intercept. This is 'bag of item inference' on yesterday's model."""
    model_old.eval(); ni = base["ni"]; headmask = base["headmask"]
    ff, tt = [], []
    Xv, metas = [], []
    for u in users:
        profset, held, prof_r, held_r = SPL[u]
        tlike = [j for j in held if held_r[j] >= LO]
        if not tlike:
            continue
        xv = np.zeros(ni, np.float32)
        if kc > 0:
            prof_items = np.array(list(prof_r.keys()), np.int64)
            prof_stars = np.array([prof_r[j] for j in prof_items], np.float64)
            cidx, g = derive_concepts(prof_items, prof_stars, cb)
            if len(cidx):
                sub = cb.Mcsc[:, prof_items]; nknown = np.asarray(sub.sum(1)).ravel()[cidx]
                sel = np.argsort(-nknown)[:kc]
                for ci, gg in zip(cidx[sel], g[sel]):
                    mem = cb.Mbin[int(ci)].indices
                    xv[mem] += gg                         # member-bag injection into ITEM channel
        Xv.append(xv); metas.append((profset, tlike))
    if not Xv:
        return float("nan"), float("nan"), 0
    with torch.no_grad():
        scores = []
        for b in range(0, len(Xv), 256):
            sc = model_old(torch.from_numpy(np.stack(Xv[b:b + 256])), dropout=0.0)[0].numpy().astype(np.float64)
            scores.append(sc)
        scores = np.concatenate(scores, 0)
    for r, (profset, tlike) in enumerate(metas):
        nf = ndcg10(scores[r], tlike, profset, headmask, False)
        nt = ndcg10(scores[r], tlike, profset, headmask, True)
        if nf is not None: ff.append(nf)
        if nt is not None: tt.append(nt)
    return (float(np.mean(ff)) if ff else float("nan"),
            float(np.mean(tt)) if tt else float("nan"), len(ff))


def eval_intercept(model, cb, base, SPL, users):
    """TURN-0 intercept: empty interview (no items, no concepts) -> the model's popularity prior NDCG@10."""
    model.eval(); ni = base["ni"]; headmask = base["headmask"]
    ff, tt = [], []
    Xv, metas = [], []
    for u in users:
        profset, held, prof_r, held_r = SPL[u]
        tlike = [j for j in held if held_r[j] >= LO]
        if not tlike:
            continue
        Xv.append(np.zeros(ni, np.float32)); metas.append((profset, tlike))
    with torch.no_grad():
        cg0 = torch.zeros((len(Xv), cb.C), dtype=torch.float32)
        scores = model(torch.from_numpy(np.stack(Xv)), cg0, cg0)[0].numpy().astype(np.float64)
    for r, (profset, tlike) in enumerate(metas):
        nf = ndcg10(scores[r], tlike, profset, headmask, False)
        nt = ndcg10(scores[r], tlike, profset, headmask, True)
        if nf is not None: ff.append(nf)
        if nt is not None: tt.append(nt)
    return (float(np.mean(ff)) if ff else float("nan"),
            float(np.mean(tt)) if tt else float("nan"), len(ff))


def cmd_cmp(args):
    """Concept elicitation comparison (single-seed val): turn-0 intercept, reconciled concept-only channel,
    and PREVIOUS a0c member-bag item-injection -- side by side, vs MOSTPOP."""
    from signed_latent import SignedAE
    base = load_arena_base(); ni = base["ni"]
    SL._POP = base["cnt"].astype(np.float64)
    cb = ConceptBank(ni, base["cnt"])
    blob = torch.load(os.path.join(OUTDIR, f"{args.tag}_best.pt"), map_location="cpu")
    model = ReconciledAE(ni, cb.C); model.load_state_dict(blob["model"]); model.eval()
    old = SignedAE(ni, use_mask=True)
    old.load_state_dict(torch.load(A03B, map_location="cpu")["model"]); old.eval()
    SPL = build_splits(base, SEEDS[0]); users = cohort(base, SPL, "val")
    # MOSTPOP
    popb = base["popb"].astype(np.float64); headmask = base["headmask"]; mp = []
    for u in users:
        profset, held, prof_r, held_r = SPL[u]
        tlike = [j for j in held if held_r[j] >= LO]
        if tlike:
            mp.append(ndcg10(popb.copy(), tlike, profset, headmask, False))
    log(f"[cmp] MOSTPOP full={np.mean([x for x in mp if x is not None]):.4f}")
    ic_f, ic_t, _ = eval_intercept(model, cb, base, SPL, users)
    log(f"[cmp] TURN-0 intercept (reconciled, empty) full={ic_f:.4f} tail={ic_t:.4f}")
    log(f"[cmp] {'kc':>3} | reconciled concept-only | a0c bag-injection")
    for kc in (0, 1, 2, 4, 8, 16):
        rf = eval_concept_only(model, cb, base, SPL, users, kc)[0] if kc > 0 else ic_f
        bf = eval_bag_injection(old, cb, base, SPL, users, kc)[0]
        log(f"[cmp] {kc:>3} | {rf:.4f}                 | {bf:.4f}")


def cmd_cc(args):
    """Concept-only-by-turn curve (single-seed peek). Reveal kc concepts (no items), NDCG@10 held-liked."""
    base = load_arena_base(); ni = base["ni"]
    SL._POP = base["cnt"].astype(np.float64)
    cb = ConceptBank(ni, base["cnt"])
    best = os.path.join(OUTDIR, f"{args.tag}_best.pt")
    blob = torch.load(best, map_location="cpu")
    model = ReconciledAE(ni, cb.C); model.load_state_dict(blob["model"]); model.eval()
    SPL = build_splits(base, SEEDS[0]); users = cohort(base, SPL, "val")
    log(f"[cc] {best} val_full={blob.get('val_full')} val_auc={blob.get('val_auc')}; concept-ONLY by turn:")
    for kc in (1, 2, 4, 8, 16, 32):
        f, t, n = eval_concept_only(model, cb, base, SPL, users, kc)
        log(f"[cc] kc={kc:2d} concepts-only NDCG@10 full={f:.4f} tail={t:.4f} (n={n})")


def cmd_gates(args):
    """FULL GATE BATTERY on a finished version's PEAK. Seed-averaged, full-catalog, no caps.
      G-strength: full-profile NDCG@10 full+tail.
      G-coldstart + A1 USEFULNESS KNOCKOUT: cold-start k-curve early-AUC with concepts ALL vs OFF vs POS-only
        vs NEG-only -> total lift, positive-half lift, negative-half lift (the dislike-useful test).
      G-sweep (supporting behavioral): monotone hated->loved + specificity.
      Hygiene: intercept (empty==prior/a0c), monotone accumulation (full >= every k)."""
    base = load_arena_base(); ni = base["ni"]
    SL._POP = base["cnt"].astype(np.float64)
    cb = ConceptBank(ni, base["cnt"])
    tag = args.tag
    best = os.path.join(OUTDIR, f"{tag}_best.pt")
    blob = torch.load(best, map_location="cpu")
    model = ReconciledAE(ni, cb.C); model.load_state_dict(blob["model"]); model.eval()
    log(f"[gates] loaded {best} (val_full={blob.get('val_full')} val_auc={blob.get('val_auc')})")
    kk = (1, 2, 4, 8)
    # accumulate seed-avg curves for each concept regime
    reg = ["all", "off", "pos", "neg"]
    sign = {"all": None, "off": None, "pos": "pos", "neg": "neg"}
    usec = {"all": True, "off": False, "pos": True, "neg": True}
    cur = {r: {k: [] for k in kk} for r in reg}
    full_f, full_t = [], []
    for seed in SEEDS:
        SPL = build_splits(base, seed); users = cohort(base, SPL, "test")
        f, t, _ = eval_strength(model, cb, base, SPL, users, "full", use_concepts=False, tail=True)
        full_f.append(f); full_t.append(t)
        for r in reg:
            for k in kk:
                cur[r][k].append(eval_strength(model, cb, base, SPL, users, "k", k=k,
                                               use_concepts=usec[r], concept_sign=sign[r])[0])
    C = {r: {k: float(np.mean(cur[r][k])) for k in kk} for r in reg}
    A = {r: _auc(C[r], kk) for r in reg}
    full = float(np.mean(full_f)); ftail = float(np.mean(full_t))
    total_lift = A["all"] - A["off"]; pos_lift = A["pos"] - A["off"]; neg_lift = A["neg"] - A["off"]
    log(f"[gates] G-strength full-profile NDCG@10 full={full:.4f} tail={ftail:.4f}  (a0c 0.4961)")
    for k in kk:
        log(f"[gates] k={k}: all={C['all'][k]:.4f} off={C['off'][k]:.4f} pos={C['pos'][k]:.4f} neg={C['neg'][k]:.4f}")
    log(f"[gates] EARLY-AUC all={A['all']:.4f} off={A['off']:.4f} | TOTAL concept lift={total_lift:+.4f} "
        f"| POS-half={pos_lift:+.4f} | NEG-half={neg_lift:+.4f}")
    log(f"[gates]   (TOTAL>0 = concepts load-bearing; NEG-half>0 = DISLIKE is useful, the prize)")
    sweep = gate_sweep(model, cb, base)
    log(f"[gates] G-sweep monotone_frac={sweep['monotone_frac']:.2f} (n={sweep['n']}) "
        f"ctrl_disp={sweep['mean_ctrl_displacement']:.3f} (low=specific)")
    # hygiene: intercept (empty interview -> item-only path == a0c prior) + monotone accumulation
    intercept_ok = abs(full - 0.4961) < 0.02
    accum_ok = all(full >= C["all"][k] - 1e-6 for k in kk)
    log(f"[gates] hygiene: intercept~a0c={intercept_ok} monotone-accum(full>=k)={accum_ok}")
    out = dict(tag=tag, strength=dict(full=full, tail=ftail),
               kcurve=C, early_auc=A,
               concept_lift=dict(total=total_lift, positive_half=pos_lift, negative_half=neg_lift),
               sweep=sweep, hygiene=dict(intercept_ok=bool(intercept_ok), accum_ok=bool(accum_ok)),
               val_full=blob.get("val_full"), val_auc=blob.get("val_auc"))
    json.dump(out, open(os.path.join(OUTDIR, f"gates_{tag}.json"), "w"), indent=2)
    log(f"[gates] wrote gates_{tag}.json")
    return out


def percentile_rank(scores, items):
    """Mean fraction-of-catalog scored BELOW each item (1.0 = top). Identical to signed_latent."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(len(scores)) / (len(scores) - 1)
    return float(np.mean(ranks[np.asarray(items, np.int64)]))


def t0_flip_check(model, cb, base, n_probe=12):
    """t=0 REPRODUCTION ASSERT (Fable A1-ii): feed a concept LOVED vs HATED via the CONCEPT channel (empty
    item input) -> its member items must RISE (loved) / FALL (hated). Direction = the -26pt probe, now
    delivered through the dedicated channel. Reports per-concept flip gap (loved_pct - hated_pct)."""
    model.eval(); ni = base["ni"]; C = cb.C
    # pick the largest-membership genome tags (clear, well-defined regions) as probes
    sizes = np.asarray((cb.Mbin.sum(1))).ravel().copy(); sizes[cb.ntag:] = 0    # tags only, well-defined
    probe_c = np.argsort(-sizes)[:n_probe]
    z0 = np.zeros(ni, np.float32)
    flips = []
    for c in probe_c:
        members = cb.Mbin[c].indices
        if len(members) < 20:
            continue
        cg = np.zeros(C, np.float32); cm = np.zeros(C, np.float32); cm[c] = 1.0
        cg[c] = VALENCE[3] * CONF[2]                                   # loved, know_well = +1
        with torch.no_grad():
            s_love = model(torch.from_numpy(z0[None]), torch.from_numpy(cg[None]),
                           torch.from_numpy(cm[None]))[0].numpy()[0].astype(np.float64)
        cg[c] = VALENCE[0] * CONF[2]                                   # hated, know_well = -1
        with torch.no_grad():
            s_hate = model(torch.from_numpy(z0[None]), torch.from_numpy(cg[None]),
                           torch.from_numpy(cm[None]))[0].numpy()[0].astype(np.float64)
        pl = percentile_rank(s_love, members); ph = percentile_rank(s_hate, members)
        flips.append(dict(concept=cb.names[c], n_members=int(len(members)),
                          pct_loved=pl, pct_hated=ph, flip_gap=pl - ph))
    mean_gap = float(np.mean([f["flip_gap"] for f in flips])) if flips else float("nan")
    return dict(mean_flip_gap=mean_gap, n_probes=len(flips), per_concept=flips)


def cmd_probe(args):
    """PRE-LAUNCH GATE (Fable A1): warm-start + encoder-faithful concept init, then ASSERT at t=0:
       (i) no-concept path reproduces a03b full-profile ~0.4961 (bit-path z=z_item);
       (ii) LOVED concept raises members / HATED lowers (the -26pt reproduction via the concept channel);
       (iii) full-profile +concepts ~= item-only (concepts correctly REDUNDANT at full coverage)."""
    base = load_arena_base(); ni = base["ni"]
    SL._POP = base["cnt"].astype(np.float64)
    cb = ConceptBank(ni, base["cnt"])
    model = ReconciledAE(ni, cb.C)
    model.init_from_a03b(cb)
    SPL = build_splits(base, SEEDS[0]); users = cohort(base, SPL, "val")
    # (i)+(iii) strength at init: item-only (== a03b) vs +concepts derived (should be ~equal: redundant)
    f_io, t_io, n = eval_strength(model, cb, base, SPL, users, "full", use_concepts=False)
    f_wc, t_wc, _ = eval_strength(model, cb, base, SPL, users, "full", use_concepts=True)
    log(f"[probe] (i) no-concept full={f_io:.4f} tail={t_io:.4f} (n={n})  [a03b target 0.4961]")
    log(f"[probe] (iii) +concepts full={f_wc:.4f} tail={t_wc:.4f}  (delta {f_wc-f_io:+.4f}; ~0 expected)")
    # (ii) t=0 flip reproduction via concept channel
    flip = t0_flip_check(model, cb, base)
    log(f"[probe] (ii) concept-channel flip: mean_gap={flip['mean_flip_gap']:+.3f} over {flip['n_probes']} concepts "
        f"(POSITIVE = loved>hated = reproduces -26pt direction)")
    reproduces_a03b = abs(f_io - 0.4961) < 0.01
    flip_ok = flip["mean_flip_gap"] > 0.05
    log(f"[probe] GATE: no-concept==a03b={reproduces_a03b}  concept-flip-directional={flip_ok}")
    out = dict(init_item_only=dict(full=f_io, tail=t_io), init_with_concepts=dict(full=f_wc, tail=t_wc),
               t0_flip=flip, gate_reproduces_a03b=bool(reproduces_a03b), gate_flip_ok=bool(flip_ok),
               C=cb.C, ntag=cb.ntag, nent=cb.nent, n_users=n)
    json.dump(out, open(os.path.join(OUTDIR, "probe_init.json"), "w"), indent=2)
    log(f"[probe] wrote probe_init.json  -- PHASE-2 GATE {'PASS' if (reproduces_a03b and flip_ok) else 'FAIL'}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["probe", "p2", "p3", "gates", "cc", "cmp", "cctrain", "utrain", "ridge", "p2ans"])
    ap.add_argument("--tag", default="recon")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.cmd == "probe":
        cmd_probe(args)
    elif args.cmd == "p2":
        cmd_p2(args)
    elif args.cmd == "gates":
        cmd_gates(args)
    elif args.cmd == "cc":
        cmd_cc(args)
    elif args.cmd == "cmp":
        cmd_cmp(args)
    elif args.cmd == "cctrain":
        cmd_cctrain(args)
    elif args.cmd == "utrain":
        cmd_utrain(args)
    elif args.cmd == "ridge":
        cmd_ridge(args)
    elif args.cmd == "p2ans":
        cmd_p2ans(args)
    else:
        raise SystemExit(f"{args.cmd} wired after Phase-2 validates")
