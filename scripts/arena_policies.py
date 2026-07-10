"""arena_policies.py -- baseline ladder + adaptive classes + interview runner (POLICY ARENA v2,
gated world; rebuilt per ARENA_CODE_AUDIT.md 2026-07-10).

AUDIT FIXES IN THIS FILE:
  #1 b4 is BLIND: objective = expected gain in the smooth ranking utility J(z)=tau*logsumexp(
     decode(z)/tau) (the AskGradient surrogate, exact not linearized) -- never the eval targets.
  #2 build_b2 RESUMES from a partial cached sequence and asserts len(seq)==Tmax.
  #3 b2 construction cohort raised (default 300 users; candidate PRE-SCREEN documented).
  #6 b3 gets a pre-registered value-ranked tail (prescreen-gain order) when the seq exhausts.
  #7 ONE shared cand_M for b4/A/B/C; all blind arms share the SAME TRAIN population prior.
  #9 deployable policies receive ONLY the observed dialogue (asked/answered/branch), never ctx;
     ctx goes to the world and the LABELLED context arms.

BASELINES: b0 cold | b1 concept-entropy static | b2 LEARNED static | b3=b2+skip+tail | b4 blind
model-based myopic (J-surrogate). CONTEXT (labelled, never cited): true-table router, clairvoyant.
ADAPTIVE: A learned scorer (GBM, 1+2-step realized-gain labels, blind features) | B ask-the-gradient
| C CAT-router (Fisher proxy) | D Golbandi polarity tree (branches on OBSERVED answers only).
Each adaptive class gets a b2-anchored tie-by-construction variant (TAU on DEV-VAL).
"""
import os, sys, json, time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.join(_HERE, "instrument2"))
import warnings
warnings.filterwarnings("ignore")

import arena_core as AC
from arena_core import (TYPE_ITEM, TYPE_CONCEPT, TYPE_ENTITY, KIND_IMPL, KIND_EXPL, LVL_ROUGH,
                        LVL_KW, FID_DATA, FID_EASE, FID_LLM, ndcg_at_k, ndcg_at_k_batch)

CACHE = AC.CACHE_DIR
SEED = AC.SEED
PRIMARY_K = 50


# ============================================================ shared BLIND machinery (E7-symmetric)
def blind_scores(arena, z):
    s = arena.FR.decode_np(z[None, :])[0]
    mu = float(np.median(s)); sd = float(np.std(s) + 1e-9)
    return s, mu, sd


def blind_value(arena, s, mu, sd, qidx):
    """Value FORECAST for a candidate region from the belief only (decode over region members)."""
    mem = arena.region_members(qidx)
    if len(mem) == 0:
        return 0.0
    mv = float(np.mean(s[mem]))
    return float(np.clip((mv - mu) / (2.0 * sd), -1.0, 1.0))


def top_answerable(arena, z, znorm, used, M):
    """Top-M unused candidates by vectorised blind answerability (shared prior, one matmul)."""
    order = np.argsort(-arena.blind_pans_all(z, znorm))
    out = []
    for qi in order:
        qi = int(qi)
        if qi not in used:
            out.append(qi)
            if len(out) >= M:
                break
    return out


def smooth_J(arena, Z, tau=0.5):
    """Smooth top-K ranking utility J(z) = tau*logsumexp(decode(z)/tau) per row (BLIND objective).
    Higher J = more concentrated high-score mass = better expected ranking under the fold."""
    S = arena.FR.decode_np(Z)                                # (B, ni)
    mx = S.max(axis=1, keepdims=True)
    return (mx[:, 0] + tau * np.log(np.exp((S - mx) / tau).sum(axis=1)))


def hypo_tokens(arena, z_scores, qidx, tokens, mu, sd):
    """Hypothetical answered-outcome tokens for a candidate (BLIND: forecast value, expected level).
    Both tokens (implicit know_well + explicit forecast value) -- the two-channel correction."""
    ch = int(arena.q_channel[qidx])
    emb = arena.Qemb[qidx]
    v = blind_value(arena, z_scores, mu, sd, qidx)
    return tokens + [(ch, KIND_IMPL, LVL_KW, 0.0, FID_DATA, 0.0, emb, 0.0),
                     (ch, KIND_EXPL, LVL_ROUGH, 0.0, FID_EASE, float(v), emb, 0.0)]


# ============================================================ the interview runner (batched)
def run_policy(arena, recs, policy, Tmax, Ks=(50, 10), skip=False, verbose=False, tag=""):
    """E1: same users, refusal = consumed turn (unless skip), belief unchanged on refusal, user kept.
    Deployable policies see ONLY the observed dialogue (fix #9); ctx goes to world + labelled arms."""
    n = len(recs)
    ctxs = [arena.user_ctx(r) for r in recs]
    held = [r["held"] for r in recs]
    prof = [set(r["known"].keys()) for r in recs]
    tokens = [[] for _ in range(n)]
    natives = [[] for _ in range(n)]
    used = [set() for _ in range(n)]
    asked = [[] for _ in range(n)]
    ansf = [[] for _ in range(n)]
    observed = [dict() for _ in range(n)]        # qi -> branch ('refuse'/'dislike'/'like'); OBSERVED
    curves = {K: np.full((n, Tmax + 1), np.nan) for K in Ks}
    privileged = getattr(policy, "privileged", False)

    policy.start(arena, recs, ctxs if privileged else None)
    static_order = policy.order_static()

    Z = arena.belief_z_batch([[] for _ in range(n)], [[] for _ in range(n)])
    for K in Ks:
        vals = ndcg_at_k_batch(arena.FR, Z, held, prof, K)
        curves[K][:, 0] = [v if v is not None else np.nan for v in vals]
    z_cur = Z

    t0 = time.time()
    for t in range(1, Tmax + 1):
        for i in range(n):
            uid = recs[i]["u"]; ctx = ctxs[i]
            while True:
                if static_order is not None:
                    qi = policy.pick_static(i, used[i], t)
                else:
                    view = ctxs[i] if privileged else observed[i]
                    qi = policy.pick(arena, i, uid, view, asked[i], ansf[i], used[i],
                                     tokens[i], natives[i], z_cur[i], t)
                if qi is None:
                    break
                used[i].add(qi)
                a = arena.answered(uid, qi, ctx)
                observed[i][qi] = arena.realized_branch(uid, qi, ctx)
                tokens[i] += arena.tokens_for(uid, qi, ctx)   # v3.1: no-clue tokens FOLD too
                if a and arena.is_liked_item(uid, qi, ctx):
                    natives[i].append(int(arena.uni.bank[qi - arena.off_item]))
                if skip and not a and len(used[i]) < arena.nQ:
                    continue                     # b3: refused turn refunded (its no-clue evidence kept)
                asked[i].append(qi); ansf[i].append(bool(a))
                break
        Z = arena.belief_z_batch(tokens, natives)
        z_cur = Z
        for K in Ks:
            vals = ndcg_at_k_batch(arena.FR, Z, held, prof, K)
            curves[K][:, t] = [v if v is not None else np.nan for v in vals]
        if verbose and t % 6 == 0:
            print(f"    [{tag}] turn {t}/{Tmax} [{time.time()-t0:.0f}s] "
                  f"NDCG@{Ks[0]}={np.nanmean(curves[Ks[0]][:, t]):.4f}", flush=True)
    return dict(curves={K: curves[K] for K in Ks}, asked=asked, ansf=ansf)


# ============================================================ base policy interface
class Policy:
    name = "base"
    privileged = False
    def start(self, arena, recs, ctxs): pass
    def order_static(self): return None
    def pick_static(self, i, used, t): return None
    def pick(self, arena, i, uid, view, asked, ansf, used, tokens, natives, z_cur, t): return None


class B0Cold(Policy):
    name = "b0_cold"
    def order_static(self): return []
    def pick_static(self, i, used, t): return None


class StaticSeq(Policy):
    """Fixed sequence; optional TAIL continuation (fix #6, for b3's refund regime)."""
    def __init__(self, name, seq, tail=None):
        self.name = name; self.seq = list(seq); self.tail = list(tail) if tail is not None else []
    def order_static(self): return self.seq
    def pick_static(self, i, used, t):
        for qi in self.seq:
            if qi not in used:
                return qi
        for qi in self.tail:
            if qi not in used:
                return qi
        return None


# ============================================================ b1: concept-entropy static
def build_b1(arena, train_recs, verbose=True):
    """Historical heuristic ported: concepts ordered by binary entropy of the TRAIN answer rate
    (ties by question index). Concept channel only (1,128)."""
    K = np.stack([arena.user_table(r["u"], r["known"])["know"] for r in train_recs])
    p = (K[:, :arena.off_ent] >= 1).mean(0)
    ent = -(p * np.log2(p + 1e-9) + (1 - p) * np.log2(1 - p + 1e-9))
    order = sorted(range(arena.off_ent), key=lambda c: (-ent[c], c))
    if verbose:
        print(f"[b1] concept-entropy static over {arena.off_ent} concepts; top-5 rates "
              f"{[round(float(p[c]), 2) for c in order[:5]]}", flush=True)
    return order


# ============================================================ TRAIN population prior (shared, blind)
def train_pop_prior(arena, train_recs):
    """Per-question TRAIN answer rate -- the ONE population prior every blind arm shares (E7)."""
    K = np.stack([arena.user_table(r["u"], r["known"])["know"] for r in train_recs])
    return (K >= 1).mean(0)


# ============================================================ b2: THE LEARNED STATIC (greedy)
def _commit_q(arena, recs, ctxs, tokens, natives, qi):
    for i, r in enumerate(recs):
        tk = arena.tokens_for(r["u"], qi, ctxs[i])            # incl no-clue tokens (v3.1)
        tokens[i] = tokens[i] + tk
        if arena.is_liked_item(r["u"], qi, ctxs[i]) and arena.answered(r["u"], qi, ctxs[i]):
            natives[i] = natives[i] + [int(arena.uni.bank[qi - arena.off_item])]


def prescreen_gains(arena, recs, K=PRIMARY_K, tag="", verbose=True):
    """1-question cohort NDCG gain for EVERY question from cold (one pass; cached). Used to (a)
    pre-screen the b2 greedy candidate set (documented compute deviation) and (b) define b3's
    value-ranked tail order."""
    path = f"{CACHE}/pres31_{tag}_n{len(recs)}_K{K}.json"
    if os.path.exists(path):
        return np.array(json.load(open(path))["gain"])
    t0 = time.time()
    ctxs = [arena.user_ctx(r) for r in recs]
    held = [r["held"] for r in recs]; prof = [set(r["known"].keys()) for r in recs]
    Z0 = arena.belief_z_batch([[] for _ in recs], [[] for _ in recs])
    base = np.array([v if v is not None else np.nan
                     for v in ndcg_at_k_batch(arena.FR, Z0, held, prof, K)])
    gains = np.zeros(arena.nQ)
    for qi in range(arena.nQ):
        tl = []; nl = []
        for i, r in enumerate(recs):
            tl.append(arena.tokens_for(r["u"], qi, ctxs[i]))
            nl.append([int(arena.uni.bank[qi - arena.off_item])]
                      if arena.is_liked_item(r["u"], qi, ctxs[i]) and
                      arena.answered(r["u"], qi, ctxs[i]) else [])
        Z = arena.belief_z_batch(tl, nl)
        nd = np.array([v if v is not None else np.nan
                       for v in ndcg_at_k_batch(arena.FR, Z, held, prof, K)])
        gains[qi] = float(np.nanmean(nd - base))
        if verbose and (qi + 1) % 400 == 0:
            print(f"    [prescreen] {qi+1}/{arena.nQ} [{time.time()-t0:.0f}s]", flush=True)
    json.dump({"gain": gains.tolist()}, open(path, "w"))
    assert os.path.exists(path)
    if verbose:
        print(f"[prescreen] done ({arena.nQ} q, {len(recs)} users) [{time.time()-t0:.0f}s]", flush=True)
    return gains


def build_b2(arena, train_recs, Tmax=24, K=PRIMARY_K, prescreen_top=300, verbose=True, tag=""):
    """Greedy forward selection on TRAIN users (E2). Candidate set = top-`prescreen_top` questions
    by 1-question cohort gain (documented deviation from argmax-over-all; the pre-screen itself
    scans ALL 2,428 on the same cohort). RESUMES from partial cache; asserts full length (fix #2)."""
    path = f"{CACHE}/b2v31_T{Tmax}_K{K}_n{len(train_recs)}{tag}.json"
    seq, gains = [], []
    if os.path.exists(path):
        blob = json.load(open(path))
        seq, gains = blob["seq"], blob["gain"]
        if len(seq) >= Tmax:
            if verbose:
                print(f"[b2] loaded COMPLETE greedy static ({len(seq)} picks) from {path}", flush=True)
            return seq[:Tmax], gains[:Tmax]
        print(f"[b2] partial cache ({len(seq)}/{Tmax}) -> RESUMING greedy (audit fix #2)", flush=True)
    t0 = time.time()
    pres = prescreen_gains(arena, train_recs, K=K, tag=f"b2{tag}", verbose=verbose)
    cands = list(np.argsort(-pres)[:prescreen_top])
    ctxs = [arena.user_ctx(r) for r in train_recs]
    held = [r["held"] for r in train_recs]; prof = [set(r["known"].keys()) for r in train_recs]
    tokens = [[] for _ in train_recs]; natives = [[] for _ in train_recs]
    for qi in seq:                                            # replay committed prefix (resume)
        _commit_q(arena, train_recs, ctxs, tokens, natives, int(qi))
    used = set(int(q) for q in seq)
    for t in range(len(seq), Tmax):
        Zc = arena.belief_z_batch(tokens, natives)
        base = np.array([v if v is not None else np.nan
                         for v in ndcg_at_k_batch(arena.FR, Zc, held, prof, K)])
        best_q, best_gain = None, -1e9
        for qi in cands:
            qi = int(qi)
            if qi in used:
                continue
            tl = []; nl = []
            for i, r in enumerate(train_recs):
                tk = arena.tokens_for(r["u"], qi, ctxs[i])
                tl.append(tokens[i] + tk)
                nl.append(natives[i] + ([int(arena.uni.bank[qi - arena.off_item])]
                          if arena.is_liked_item(r["u"], qi, ctxs[i]) and
                          arena.answered(r["u"], qi, ctxs[i]) else []))
            Z = arena.belief_z_batch(tl, nl)
            nd = np.array([v if v is not None else np.nan
                           for v in ndcg_at_k_batch(arena.FR, Z, held, prof, K)])
            g = float(np.nanmean(nd - base))
            if g > best_gain:
                best_gain = g; best_q = qi
        used.add(best_q); seq.append(int(best_q)); gains.append(best_gain)
        _commit_q(arena, train_recs, ctxs, tokens, natives, best_q)
        if verbose:
            print(f"[b2] t={t+1:2d} pick {arena.q_names[best_q][:44]:44s} gain {best_gain:+.4f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        json.dump({"seq": seq, "gain": gains}, open(path, "w"))
    assert len(seq) >= Tmax, f"b2 incomplete: {len(seq)}/{Tmax}"
    assert os.path.exists(path)
    return seq[:Tmax], gains[:Tmax]


def b3_tail(arena, train_recs, b2_seq, tag=""):
    """Pre-registered b3 continuation: remaining universe by prescreen 1-question gain, descending."""
    pres = prescreen_gains(arena, train_recs, tag=f"b2{tag}", verbose=False)
    inseq = set(int(q) for q in b2_seq)
    return [int(q) for q in np.argsort(-pres) if int(q) not in inseq]


# ============================================================ b4: BLIND model-based myopic greedy
class B4Myopic(Policy):
    """Fix #1: fully BLIND deployable myopic greedy. Each turn: over the shared candidate subset,
    expected gain = p_ans_blind(q) * (J(z') - J(z)) with J = tau*logsumexp(decode/tau) (the smooth
    ranking utility; the AskGradient surrogate computed EXACTLY, not linearized) and z' = fold of
    the hypothetical blind-forecast answered outcome. Never touches held-out targets."""
    name = "b4_myopic"
    def __init__(self, M=100, tau=0.5):
        self.M = M; self.tau = tau
    def pick(self, arena, i, uid, view, asked, ansf, used, tokens, natives, z_cur, t):
        s, mu, sd = blind_scores(arena, z_cur)
        znorm = float(np.linalg.norm(z_cur) + 1e-9)
        cand = top_answerable(arena, z_cur, znorm, used, self.M)
        if not cand:
            return None
        pans = arena.blind_pans_all(z_cur, znorm)
        tl = [hypo_tokens(arena, s, qi, tokens, mu, sd) for qi in cand]
        Z = arena.belief_z_batch(tl, [list(natives)] * len(tl))
        J1 = smooth_J(arena, Z, self.tau)
        J0 = smooth_J(arena, z_cur[None, :], self.tau)[0]
        eg = pans[cand] * (J1 - J0)
        return cand[int(np.argmax(eg))]


# ============================================================ context arms (LABELLED, never cited)
class TrueTableRouter(Policy):
    """CONTEXT (privileged, labelled): per-turn argmax of REALIZED 1-step NDCG@K gain."""
    name = "ctx_truetable"
    privileged = True
    def __init__(self, M=100, K=PRIMARY_K):
        self.M = M; self.K = K
    def start(self, arena, recs, ctxs):
        self.held = [r["held"] for r in recs]; self.prof = [set(r["known"].keys()) for r in recs]
    def pick(self, arena, i, uid, ctx, asked, ansf, used, tokens, natives, z_cur, t):
        held = self.held[i]; prof = self.prof[i]
        znorm = float(np.linalg.norm(z_cur) + 1e-9)
        cand = top_answerable(arena, z_cur, znorm, used, self.M)
        if not cand:
            return None
        tl = []; nl = []
        for qi in cand:
            tk = arena.tokens_for(uid, qi, ctx)
            tl.append(tokens + tk)
            nl.append(natives + ([int(arena.uni.bank[qi - arena.off_item])]
                      if arena.is_liked_item(uid, qi, ctx) and arena.answered(uid, qi, ctx) else []))
        Z = arena.belief_z_batch(tl, nl)
        nd = ndcg_at_k_batch(arena.FR, Z, [held] * len(tl), [prof] * len(tl), self.K)
        return cand[int(np.nanargmax([v if v is not None else -1 for v in nd]))]


class Clairvoyant(Policy):
    """CONTEXT (privileged, labelled, APPROXIMATE ceiling): realized greedy over the M most-
    answerable ANSWERED candidates."""
    name = "ctx_clairvoyant"
    privileged = True
    def __init__(self, K=PRIMARY_K, M=200):
        self.K = K; self.M = M
    def start(self, arena, recs, ctxs):
        self.held = [r["held"] for r in recs]; self.prof = [set(r["known"].keys()) for r in recs]
    def pick(self, arena, i, uid, ctx, asked, ansf, used, tokens, natives, z_cur, t):
        held = self.held[i]; prof = self.prof[i]
        know = ctx["table"]["know"]
        cand = [qi for qi in range(arena.nQ) if qi not in used and know[qi] >= 1]
        if not cand:
            cand = [qi for qi in range(arena.nQ) if qi not in used]
        if len(cand) > self.M:
            znorm = float(np.linalg.norm(z_cur) + 1e-9)
            pans = arena.blind_pans_all(z_cur, znorm)
            cand = sorted(cand, key=lambda qi: -pans[qi])[:self.M]
        tl = []; nl = []
        for qi in cand:
            tk = arena.tokens_for(uid, qi, ctx)
            tl.append(tokens + tk)
            nl.append(natives + ([int(arena.uni.bank[qi - arena.off_item])]
                      if arena.is_liked_item(uid, qi, ctx) and arena.answered(uid, qi, ctx) else []))
        Z = arena.belief_z_batch(tl, nl)
        nd = ndcg_at_k_batch(arena.FR, Z, [held] * len(tl), [prof] * len(tl), self.K)
        return cand[int(np.nanargmax([v if v is not None else -1 for v in nd]))]


# ============================================================ B: ask-the-gradient (blind)
class AskGradient(Policy):
    """direction g = dJ/dz (softmax-weighted decoder rows); score(q) = p_ans * (E[dz] . g);
    snap = argmax within predicted-answerable territory. Blind (forecast value, shared prior)."""
    name = "B_gradient"
    def __init__(self, tau=0.5, floor=0.35, M=100):
        self.tau = tau; self.floor = floor; self.M = M
    def start(self, arena, recs, ctxs):
        self.W = arena.FR.W.numpy().astype(np.float64)
    def _direction(self, arena, z):
        s = arena.FR.decode_np(z[None, :])[0]
        s = s - s.max()
        w = np.exp(s / self.tau); w /= w.sum()
        idx = np.argpartition(-w, 2000)[:2000]
        g = (w[idx][:, None] * self.W[idx]).sum(0)
        return g / (np.linalg.norm(g) + 1e-9)
    def pick(self, arena, i, uid, view, asked, ansf, used, tokens, natives, z_cur, t):
        g = self._direction(arena, z_cur)
        s, mu, sd = blind_scores(arena, z_cur)
        znorm = float(np.linalg.norm(z_cur) + 1e-9)
        cand = top_answerable(arena, z_cur, znorm, used, self.M)
        if not cand:
            return None
        pans = arena.blind_pans_all(z_cur, znorm)
        keep = [qi for qi in cand if pans[qi] >= self.floor] or cand
        tl = [hypo_tokens(arena, s, qi, tokens, mu, sd) for qi in keep]
        Z = arena.belief_z_batch(tl, [list(natives)] * len(tl))
        proj = (Z - z_cur[None, :]) @ g
        sc = pans[keep] * proj
        return keep[int(np.argmax(sc))]


# ============================================================ C: CAT-router (Fisher proxy, blind)
class CATRouter(Policy):
    """Max-information: p_ans * ||E[dz]||^2 (answerable AND belief-moving). Blind."""
    name = "C_cat"
    def __init__(self, floor=0.30, M=100):
        self.floor = floor; self.M = M
    def pick(self, arena, i, uid, view, asked, ansf, used, tokens, natives, z_cur, t):
        s, mu, sd = blind_scores(arena, z_cur)
        znorm = float(np.linalg.norm(z_cur) + 1e-9)
        cand = top_answerable(arena, z_cur, znorm, used, self.M)
        if not cand:
            return None
        pans = arena.blind_pans_all(z_cur, znorm)
        keep = [qi for qi in cand if pans[qi] >= self.floor] or cand
        tl = [hypo_tokens(arena, s, qi, tokens, mu, sd) for qi in keep]
        Z = arena.belief_z_batch(tl, [list(natives)] * len(tl))
        dz = Z - z_cur[None, :]
        info = pans[keep] * (dz ** 2).sum(axis=1)
        return keep[int(np.argmax(info))]


# ============================================================ D: Golbandi polarity tree
class GolbandiTree(Policy):
    """Golbandi-2011-class conditional tree grown on TRAIN users; branches on the user's OBSERVED
    answer polarity (refuse/dislike/like) of already-asked questions -- observed dialogue only
    (fix #9). After the tree: global b2 order, then the value-ranked tail."""
    name = "D_golbandi"
    def __init__(self, tree, b2_seq, tail):
        self.tree = tree; self.tail = list(b2_seq) + list(tail)
    def pick(self, arena, i, uid, view, asked, ansf, used, tokens, natives, z_cur, t):
        node = self.tree
        while isinstance(node, dict) and node.get("q") is not None:
            qi = int(node["q"])
            if qi not in used:
                return qi
            br = view.get(qi, "like")                         # observed branch of the asked q
            node = node.get(br) or node.get("like")
        for qi in self.tail:
            if qi not in used:
                return qi
        for qi in range(arena.nQ):
            if qi not in used:
                return qi
        return None


def build_golbandi(arena, train_recs, max_depth=5, min_users=25, K=PRIMARY_K, cand_cap=250,
                   verbose=True, tag=""):
    """Tree growth on TRAIN (world-side construction; realized answers are training data -- E2).
    Split candidates = top-`cand_cap` by TRAIN answer rate (documented)."""
    path = f"{CACHE}/golbandi31_d{max_depth}_n{len(train_recs)}{tag}.json"
    if os.path.exists(path):
        if verbose:
            print(f"[D] loaded cached Golbandi tree from {path}", flush=True)
        return json.load(open(path))["tree"]
    t0 = time.time()
    ctxs = {r["u"]: arena.user_ctx(r) for r in train_recs}
    held = {r["u"]: r["held"] for r in train_recs}
    prof = {r["u"]: set(r["known"].keys()) for r in train_recs}
    Kt = np.stack([arena.user_table(r["u"], r["known"])["know"] for r in train_recs])
    ans_rate = (Kt >= 1).mean(0)
    cand_pool = [int(q) for q in np.argsort(-ans_rate)[:cand_cap]]

    def user_tokens(uid, path_q):
        toks = []; nat = []; ctx = ctxs[uid]
        for qi in path_q:
            toks += arena.tokens_for(uid, qi, ctx)            # incl no-clue tokens (v3.1)
            if arena.answered(uid, qi, ctx) and arena.is_liked_item(uid, qi, ctx):
                nat.append(int(arena.uni.bank[qi - arena.off_item]))
        return toks, nat

    def grow(uids, path_q, depth):
        if depth >= max_depth or len(uids) < min_users:
            return {"leaf": len(uids)}
        base_tok = {u: user_tokens(u, path_q) for u in uids}
        best_q, best_val, best_split = None, -1e9, None
        for qi in cand_pool:
            if qi in path_q:
                continue
            groups = {"refuse": [], "dislike": [], "like": []}
            for u in uids:
                groups[arena.realized_branch(u, int(qi), ctxs[u])].append(u)
            if max(len(groups["dislike"]), len(groups["like"])) < min_users:
                continue
            tl = []; nl = []; order = []
            for u in uids:
                tk = arena.tokens_for(u, int(qi), ctxs[u])
                tl.append(base_tok[u][0] + tk)
                nl.append(base_tok[u][1] + ([int(arena.uni.bank[qi - arena.off_item])]
                          if arena.is_liked_item(u, int(qi), ctxs[u]) and
                          arena.answered(u, int(qi), ctxs[u]) else []))
                order.append(u)
            Z = arena.belief_z_batch(tl, nl)
            nd = ndcg_at_k_batch(arena.FR, Z, [held[u] for u in order], [prof[u] for u in order], K)
            v = float(np.nanmean([x if x is not None else np.nan for x in nd]))
            if v > best_val:
                best_val = v; best_q = int(qi); best_split = groups
        if best_q is None:
            return {"leaf": len(uids)}
        node = {"q": best_q}
        if verbose:
            print(f"[D] depth {depth} split {arena.q_names[best_q][:38]:38s} "
                  f"n={[len(best_split[b]) for b in ('refuse','dislike','like')]} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        for br in ("refuse", "dislike", "like"):
            node[br] = grow(best_split[br], path_q + [best_q], depth + 1) if best_split[br] \
                else {"leaf": 0}
        return node

    tree = grow([r["u"] for r in train_recs], [], 0)
    json.dump({"tree": tree}, open(path, "w"))
    assert os.path.exists(path)
    if verbose:
        print(f"[D] tree grown (depth<={max_depth}) [{time.time()-t0:.0f}s]", flush=True)
    return tree


# ============================================================ A: learned scorer v2 (GBM, blind)
def _feat(arena, qi, z_cur, s, mu, sd, znorm, pans, n_ans, n_ref, t, Tmax, horizon):
    ch = int(arena.q_channel[qi])
    emb = arena.Qemb[qi].astype(np.float64)
    align = float(emb @ z_cur) / (znorm * (np.linalg.norm(emb) + 1e-9)) if znorm > 1e-9 else 0.0
    vmag = abs(blind_value(arena, s, mu, sd, qi))
    return [1.0 if ch == TYPE_ITEM else 0.0, 1.0 if ch == TYPE_CONCEPT else 0.0,
            1.0 if ch == TYPE_ENTITY else 0.0, float(pans[qi]), align, znorm, vmag,
            float(n_ans), float(n_ref), t / Tmax, float(horizon)]


FEAT_NAMES = ["is_item", "is_concept", "is_entity", "p_ans_blind", "align", "znorm", "vmag_blind",
              "n_ans", "n_ref", "turn_frac", "horizon"]


def build_scorerA(arena, train_recs, n_samples=4000, M2=12, Tmax=24, K=PRIMARY_K, verbose=True,
                  seed=None, tag=""):
    """Outcome-labelled scorer: BLIND features -> realized 1-step AND 2-step NDCG gains from
    simulated interviews on TRAIN users (world-side labels; E2). HistGBM."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    import joblib
    path = f"{CACHE}/scorerA31_n{len(train_recs)}_s{n_samples}{tag}.joblib"
    if os.path.exists(path):
        if verbose:
            print(f"[A] loaded cached GBM from {path}", flush=True)
        return joblib.load(path)
    t0 = time.time()
    rng = np.random.default_rng(SEED if seed is None else seed)
    ctxs = {r["u"]: arena.user_ctx(r) for r in train_recs}
    X, Y = [], []
    for si in range(n_samples):
        r = train_recs[rng.integers(len(train_recs))]; uid = r["u"]; ctx = ctxs[uid]
        held = r["held"]; prof = set(r["known"].keys())
        plen = int(rng.integers(0, 9)); used = set(); toks = []; nat = []; n_ans = 0; n_ref = 0
        allq = rng.permutation(arena.nQ); pi = 0
        while len(used) < plen and pi < len(allq):
            qi = int(allq[pi]); pi += 1; used.add(qi)
            toks += arena.tokens_for(uid, qi, ctx)            # incl no-clue tokens (v3.1)
            if arena.answered(uid, qi, ctx):
                n_ans += 1
                if arena.is_liked_item(uid, qi, ctx):
                    nat.append(int(arena.uni.bank[qi - arena.off_item]))
            else:
                n_ref += 1
        z_cur = arena.belief_z_batch([toks], [nat])[0]
        s, mu, sd = blind_scores(arena, z_cur); znorm = float(np.linalg.norm(z_cur) + 1e-9)
        pans = arena.blind_pans_all(z_cur, znorm)
        base = ndcg_at_k(arena.FR, z_cur, held, prof, K); base = 0.0 if base is None else base
        cpool = top_answerable(arena, z_cur, znorm, used, 80)
        qi = int(cpool[rng.integers(len(cpool))])
        tk = arena.tokens_for(uid, qi, ctx)
        nv = nat + ([int(arena.uni.bank[qi - arena.off_item])]
                    if arena.is_liked_item(uid, qi, ctx) and arena.answered(uid, qi, ctx) else [])
        z1 = arena.belief_z_batch([toks + tk], [nv])[0]
        n1 = ndcg_at_k(arena.FR, z1, held, prof, K); n1 = base if n1 is None else n1
        g1 = n1 - base
        X.append(_feat(arena, qi, z_cur, s, mu, sd, znorm, pans, n_ans, n_ref, len(used), Tmax, 1))
        Y.append(g1)
        used2 = used | {qi}
        c2 = top_answerable(arena, z_cur, znorm, used2, M2)
        tl = []; nl = []
        for q2 in c2:
            tk2 = arena.tokens_for(uid, q2, ctx)
            tl.append(toks + tk + tk2)
            nl.append(nv + ([int(arena.uni.bank[q2 - arena.off_item])]
                      if arena.is_liked_item(uid, q2, ctx) and arena.answered(uid, q2, ctx) else []))
        if tl:
            Z2 = arena.belief_z_batch(tl, nl)
            nd2 = ndcg_at_k_batch(arena.FR, Z2, [held] * len(tl), [prof] * len(tl), K)
            g2 = max((v if v is not None else base) for v in nd2) - base
        else:
            g2 = g1
        X.append(_feat(arena, qi, z_cur, s, mu, sd, znorm, pans, n_ans, n_ref, len(used), Tmax, 2))
        Y.append(g2)
        if verbose and (si + 1) % 1000 == 0:
            print(f"    [A] labels {si+1}/{n_samples} [{time.time()-t0:.0f}s]", flush=True)
    X = np.array(X); Y = np.array(Y)
    gbm = HistGradientBoostingRegressor(max_iter=300, max_depth=4, learning_rate=0.06,
                                        min_samples_leaf=40,
                                        random_state=SEED if seed is None else seed)
    gbm.fit(X, Y)
    joblib.dump(gbm, path)
    assert os.path.exists(path)
    if verbose:
        print(f"[A] GBM trained on {len(X)} rows (train R^2 {gbm.score(X, Y):.3f}) -> {path} "
              f"[{time.time()-t0:.0f}s]", flush=True)
    return gbm


class ScorerA(Policy):
    name = "A_scorer"
    def __init__(self, gbm, M=100, Tmax=24):
        self.gbm = gbm; self.M = M; self.Tmax = Tmax
    def pick(self, arena, i, uid, view, asked, ansf, used, tokens, natives, z_cur, t):
        n_ans = int(sum(ansf)); n_ref = int(len(ansf) - n_ans)
        s, mu, sd = blind_scores(arena, z_cur); znorm = float(np.linalg.norm(z_cur) + 1e-9)
        pans = arena.blind_pans_all(z_cur, znorm)
        cand = top_answerable(arena, z_cur, znorm, used, self.M)
        if not cand:
            return None
        X = np.array([_feat(arena, qi, z_cur, s, mu, sd, znorm, pans, n_ans, n_ref, t,
                            self.Tmax, 2) for qi in cand])
        return cand[int(np.argmax(self.gbm.predict(X)))]


# ============================================================ b2-anchored wrapper
class B2Anchored(Policy):
    """b2's order for the first TAU turns, then the inner policy. TAU=Tmax == b2 exactly."""
    def __init__(self, name, b2_seq, inner, tau):
        self.name = name; self.b2 = list(b2_seq); self.inner = inner; self.tau = tau
        self.privileged = getattr(inner, "privileged", False)
    def start(self, arena, recs, ctxs):
        self.inner.start(arena, recs, ctxs)
    def pick(self, arena, i, uid, view, asked, ansf, used, tokens, natives, z_cur, t):
        if t <= self.tau:
            for qi in self.b2:
                if qi not in used:
                    return qi
            return None
        return self.inner.pick(arena, i, uid, view, asked, ansf, used, tokens, natives, z_cur, t)
