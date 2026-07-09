"""arena_policies.py -- the baseline ladder + adaptive classes + interview runner (POLICY ARENA).

All arms per DESIGN_SHEET_POLICY_ARENA.md, E1-E7 discipline (see arena_core header).

BASELINES:  b0 cold | b1 concept-entropy static | b2 LEARNED static (greedy on TRAIN, full universe) |
            b3 = b2+skip | b4 model-based myopic greedy (population answerer model, no learning).
CONTEXT (labelled, never cited): true-table router (ttab), clairvoyant (clair).
ADAPTIVE:   A learned-scorer-v2 (GBM on 1-step+2-step gain labels) | B ask-the-gradient |
            C CAT-router (Fisher-info proxy) | D Golbandi ternary tree.
Each adaptive class also gets a b2-ANCHORED tie-by-construction variant: follow b2's order for the
first TAU turns then switch to the policy; TAU selected on DEV-VAL (TAU=Tmax => exactly b2 = tie).

Runner is batched across users per turn. NO LLM calls. Deterministic. Checkpoints to .cache/arena/.
"""
import os, sys, json, time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.join(_HERE, "instrument2"))
import warnings
warnings.filterwarnings("ignore")

import arena_core as AC
from arena_core import (TYPE_ITEM, TYPE_CONCEPT, TYPE_ATTR, TYPE_ENTITY, LVL_NEG, LVL_ROUGH, LVL_KW,
                        KIND_IMPL, KIND_EXPL, FID_DATA, FID_EASE, FID_LLM, CH_NAME, ndcg_at_k_batch,
                        ndcg_at_k)
import i25_fold_v3_sampler as SP

CACHE = AC.CACHE_DIR
SEED = AC.SEED
PRIMARY_K = 50


# ============================================================ answerer-model priors (population, blind)
def p_ans_pop(arena, ch, surp):
    """Population (trait=0) answerability under the v2.1 knowledge model -- the DEPLOYABLE prior."""
    s = float(np.clip(surp, -SP.SURPRISE_CLIP, SP.SURPRISE_CLIP))
    return SP._sig(SP._logit(SP.BASE_ANS[ch]) + SP.K_ANS * s)


def p_kw_pop(arena, ch, surp):
    s = float(np.clip(surp, -SP.SURPRISE_CLIP, SP.SURPRISE_CLIP))
    return SP._sig(SP._logit(SP.BASE_KW[ch]) + SP.K_KW * s)


# ============================================================ BLIND estimators (deployability: the
# agent starts cold and may NOT peek at the user's hidden profile -- STATE 3.1/3.3, E3). All adaptive
# arms CHOOSE questions using ONLY the current belief + population channel priors; they OBSERVE the
# true answer only AFTER asking (run_policy applies the realized token). True surprise/value are used
# solely by the WORLD and by the LABELLED context arms.
def blind_scores(arena, z):
    """Decode the current belief to per-item scores + robust location/scale for value forecasting."""
    s = arena.FR.decode_np(z[None, :])[0]
    mu = float(np.median(s)); sd = float(np.std(s) + 1e-9)
    return s, mu, sd


def blind_value(arena, s, mu, sd, qidx):
    """Predicted centered value token for a candidate region, forecast from the belief ONLY."""
    mem = arena.region_members(qidx)
    if len(mem) == 0:
        return 0.0
    mv = float(np.mean(s[mem]))
    return float(np.clip((mv - mu) / (2.0 * sd), -1.0, 1.0))


def top_answerable(arena, z, znorm, used, M):
    """Top-M unused candidates by VECTORISED blind answerability (one matmul, not a 1530-loop)."""
    order = np.argsort(-arena.blind_pans_all(z, znorm))
    out = []
    for qi in order:
        qi = int(qi)
        if qi not in used:
            out.append(qi)
            if len(out) >= M:
                break
    return out


def blind_pans(arena, z, znorm, qidx):
    """Blind answerability estimate: channel population base rate, bumped by belief-region alignment
    (regions the belief thinks the user engaged are likelier answerable). At cold start (z~0) this is
    exactly the channel prior (item .95 / concept .74 / attr .82 / entity .55)."""
    ch = arena.Q[qidx][0]
    emb = arena.q_emb(qidx).astype(np.float64)
    align = float(emb @ z) / (znorm * (np.linalg.norm(emb) + 1e-9)) if znorm > 1e-9 else 0.0
    return SP._sig(SP._logit(SP.BASE_ANS[ch]) + 1.5 * align)


# ============================================================ the interview runner (batched)
def run_policy(arena, recs, policy, Tmax, Ks=(50, 10), skip=False, verbose=False, tag=""):
    """Run `policy` for every user to horizon Tmax. Returns dict with per-user-per-turn NDCG for each
    K (n, Tmax+1) incl. turn 0 (cold), plus asked qidx / answered flags per user (mechanism)."""
    n = len(recs)
    ctxs = [arena.user_ctx(r) for r in recs]
    held = [r["held"] for r in recs]
    prof = [set(r["known"].keys()) for r in recs]
    tokens = [[] for _ in range(n)]
    natives = [[] for _ in range(n)]
    used = [set() for _ in range(n)]
    asked = [[] for _ in range(n)]                 # qidx that COUNTED as a turn (answered, or refused if !skip)
    ansf = [[] for _ in range(n)]                  # answered? per counted turn
    curves = {K: np.full((n, Tmax + 1), np.nan) for K in Ks}

    policy.start(arena, recs, ctxs)
    static_order = policy.order_static()

    # turn 0 (cold)
    Z = arena.belief_z_batch([[] for _ in range(n)], [[] for _ in range(n)])
    for K in Ks:
        vals = ndcg_at_k_batch(arena.FR, Z, held, prof, K)
        curves[K][:, 0] = [v if v is not None else np.nan for v in vals]
    z_cur = Z

    t0 = time.time()
    for t in range(1, Tmax + 1):
        for i in range(n):
            uid = recs[i]["u"]; ctx = ctxs[i]
            # choose next question (skip: keep drawing until answered or exhausted)
            while True:
                if static_order is not None:
                    qi = policy.pick_static(i, used[i], t)
                else:
                    qi = policy.pick(arena, i, uid, ctx, asked[i], ansf[i], used[i],
                                     tokens[i], natives[i], z_cur[i], t)
                if qi is None:
                    break
                used[i].add(qi)
                a = arena.answered(uid, qi, ctx)
                if skip and not a and len(used[i]) < arena.nQ:
                    continue                        # refunded: don't count, try next
                asked[i].append(qi); ansf[i].append(bool(a))
                if a:
                    tokens[i] += arena.tokens_for(uid, qi, ctx)
                    if arena.is_liked_item(uid, qi, ctx):
                        natives[i].append(int(arena.Q[qi][1]))
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
    def start(self, arena, recs, ctxs): pass
    def order_static(self): return None
    def pick_static(self, i, used, t): return None
    def pick(self, arena, i, uid, ctx, asked, ansf, used, tokens, natives, z_cur, t): return None


# ---- b0 cold -------------------------------------------------------------------------------------
class B0Cold(Policy):
    name = "b0_cold"
    def order_static(self): return []
    def pick_static(self, i, used, t): return None   # never asks -> belief stays cold


# ---- static-sequence policy (b1, b2, b3 share this) ---------------------------------------------
class StaticSeq(Policy):
    def __init__(self, name, seq):
        self.name = name; self.seq = list(seq)
    def order_static(self): return self.seq
    def pick_static(self, i, used, t):
        for qi in self.seq:
            if qi not in used:
                return qi
        return None


# ============================================================ b1: concept-entropy static
def build_b1(arena, train_recs, verbose=True):
    """Port of the historical concept-entropy heuristic: order concepts by binary entropy of the
    population answer-rate (max at p=0.5 = most informative), ties by qidx. Concept channel only."""
    concept_idx = arena.idx_by_channel[TYPE_CONCEPT]
    # population answer-rate per concept over TRAIN users
    ctxs = [arena.user_ctx(r) for r in train_recs]
    p = np.zeros(len(concept_idx))
    for k, qi in enumerate(concept_idx):
        ans = 0
        for r, ctx in zip(train_recs, ctxs):
            if arena.answered(r["u"], int(qi), ctx):
                ans += 1
        p[k] = ans / max(len(train_recs), 1)
    ent = -(p * np.log2(p + 1e-9) + (1 - p) * np.log2(1 - p + 1e-9))
    order = sorted(range(len(concept_idx)), key=lambda k: (-ent[k], int(concept_idx[k])))
    seq = [int(concept_idx[k]) for k in order]
    if verbose:
        print(f"[b1] concept-entropy static: {len(seq)} concepts; top-5 answer-rates "
              f"{[round(float(p[k]),2) for k in order[:5]]}", flush=True)
    return seq


# ============================================================ b2: THE LEARNED STATIC (greedy)
def build_b2(arena, train_recs, Tmax=24, K=PRIMARY_K, cand_cap=None, verbose=True):
    """Greedy forward selection over the FULL universe on TRAIN synthetic users, maximising mean
    NDCG@K. E7-symmetric: trained on the same synthetic world as the policies (no test-fit leak).
    Cached to disk. Returns (seq, per_step_gain)."""
    path = f"{CACHE}/b2_seq_T{Tmax}_K{K}_n{len(train_recs)}.json"
    if os.path.exists(path):
        blob = json.load(open(path))
        if verbose:
            print(f"[b2] loaded cached greedy static ({len(blob['seq'])} q) from {path}", flush=True)
        return blob["seq"], blob["gain"]
    t0 = time.time()
    n = len(train_recs)
    ctxs = [arena.user_ctx(r) for r in train_recs]
    held = [r["held"] for r in train_recs]
    prof = [set(r["known"].keys()) for r in train_recs]
    tokens = [[] for _ in range(n)]
    natives = [[] for _ in range(n)]
    seq = []; gains = []; used = set()
    # candidate list (full universe); optional cap by popularity-ish for speed (documented)
    cands = list(range(arena.nQ))
    for t in range(Tmax):
        # current mean NDCG
        Zc = arena.belief_z_batch(tokens, natives)
        base = np.array([v if v is not None else np.nan
                         for v in ndcg_at_k_batch(arena.FR, Zc, held, prof, K)])
        best_q, best_gain = None, -1e9
        # evaluate each candidate: append its (per-user) tokens, batched fold + decode
        for qi in cands:
            if qi in used:
                continue
            tl = []; nl = []
            for i in range(n):
                tk = arena.tokens_for(train_recs[i]["u"], qi, ctxs[i])
                tl.append(tokens[i] + tk)
                nvi = natives[i]
                if arena.is_liked_item(train_recs[i]["u"], qi, ctxs[i]):
                    nvi = natives[i] + [int(arena.Q[qi][1])]
                nl.append(nvi)
            Z = arena.belief_z_batch(tl, nl)
            nd = np.array([v if v is not None else np.nan
                           for v in ndcg_at_k_batch(arena.FR, Z, held, prof, K)])
            g = float(np.nanmean(nd - base))
            if g > best_gain:
                best_gain = g; best_q = qi
        used.add(best_q); seq.append(int(best_q)); gains.append(best_gain)
        # commit best_q
        for i in range(n):
            tk = arena.tokens_for(train_recs[i]["u"], best_q, ctxs[i])
            tokens[i] += tk
            if arena.is_liked_item(train_recs[i]["u"], best_q, ctxs[i]):
                natives[i].append(int(arena.Q[best_q][1]))
        if verbose:
            print(f"[b2] t={t+1:2d} pick {arena.q_names[best_q][:42]:42s} gain {best_gain:+.4f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        json.dump({"seq": seq, "gain": gains}, open(path, "w"), indent=1)
    assert os.path.exists(path)
    return seq, gains


# ============================================================ b4: model-based myopic greedy
class B4Myopic(Policy):
    """Deployable, no learning: each turn pick argmax over a candidate subset of the EXPECTED 1-step
    NDCG@K gain under the SAME answerer model (population, trait=0) and fold. Candidate subset = top-M
    unused by population answerability (documented scale-down of the argmax-over-all-questions ideal)."""
    name = "b4_myopic"
    def __init__(self, M=120, K=PRIMARY_K):
        self.M = M; self.K = K
    def start(self, arena, recs, ctxs):
        self.held = [r["held"] for r in recs]
        self.prof = [set(r["known"].keys()) for r in recs]

    def pick(self, arena, i, uid, ctx, asked, ansf, used, tokens, natives, z_cur, t):
        held = self.held[i]; prof = self.prof[i]
        if not held:
            return None
        s, mu, sd = blind_scores(arena, z_cur)
        znorm = float(np.linalg.norm(z_cur) + 1e-9)
        # candidate subset: top-M unused by BLIND answerability (belief + channel prior)
        cand = top_answerable(arena, z_cur, znorm, used, self.M)
        if not cand:
            return None
        base = ndcg_at_k(arena.FR, z_cur, held, prof, self.K); base = 0.0 if base is None else base
        # expected gain per candidate: outcome {refuse:0, answer}; value FORECAST from belief (blind)
        tl = []; meta = []
        for qi in cand:
            ch, key = arena.Q[qi]
            pa = blind_pans(arena, z_cur, znorm, qi)
            emb = arena.q_emb(qi)
            v = blind_value(arena, s, mu, sd, qi)
            toks = list(tokens) + [(ch, KIND_IMPL, LVL_KW, 0.0, FID_DATA, 0.0, emb),
                                   (ch, KIND_EXPL, LVL_ROUGH, 0.0, FID_EASE, float(v), emb)]
            tl.append(toks); meta.append((qi, pa))
        Z = arena.belief_z_batch(tl, [list(natives)] * len(tl))
        nd = ndcg_at_k_batch(arena.FR, Z, [held] * len(tl), [prof] * len(tl), self.K)
        exp_gain = {}
        for (qi, pa), v in zip(meta, nd):
            exp_gain[qi] = pa * ((0.0 if v is None else v) - base)
        return max(exp_gain, key=exp_gain.get)


# ============================================================ context arms (LABELLED, never cited)
class TrueTableRouter(Policy):
    """CONTEXT (privileged, labelled): per-turn argmax of REALIZED 1-step NDCG@K gain -- peeks the
    user's actual answer. Upper bound on the myopic policy class; NEVER a results row."""
    name = "ctx_truetable"
    def __init__(self, M=120, K=PRIMARY_K):
        self.M = M; self.K = K
    def start(self, arena, recs, ctxs):
        self.held = [r["held"] for r in recs]; self.prof = [set(r["known"].keys()) for r in recs]
    def pick(self, arena, i, uid, ctx, asked, ansf, used, tokens, natives, z_cur, t):
        held = self.held[i]; prof = self.prof[i]
        if not held:
            return None
        cand = [qi for qi in range(arena.nQ) if qi not in used]
        # subset by answerability for tractability
        scored = sorted(cand, key=lambda qi: -p_ans_pop(arena, arena.Q[qi][0],
                        arena._answer_raw(uid, qi, ctx)[3]))[:self.M]
        tl = []; nl = []
        for qi in scored:
            tk = arena.tokens_for(uid, qi, ctx)
            tl.append(tokens + tk)
            nv = natives + ([int(arena.Q[qi][1])] if arena.is_liked_item(uid, qi, ctx) else [])
            nl.append(nv)
        Z = arena.belief_z_batch(tl, nl)
        nd = ndcg_at_k_batch(arena.FR, Z, [held] * len(tl), [prof] * len(tl), self.K)
        best = int(np.nanargmax([v if v is not None else -1 for v in nd]))
        return scored[best]


class Clairvoyant(Policy):
    """CONTEXT (privileged): greedy on realized gain over ALL questions (no answerability subset).
    The ceiling. NEVER a results row."""
    name = "ctx_clairvoyant"
    def __init__(self, K=PRIMARY_K, M=200):
        self.K = K; self.M = M
    def start(self, arena, recs, ctxs):
        self.held = [r["held"] for r in recs]; self.prof = [set(r["known"].keys()) for r in recs]
    def pick(self, arena, i, uid, ctx, asked, ansf, used, tokens, natives, z_cur, t):
        held = self.held[i]; prof = self.prof[i]
        if not held:
            return None
        cand = [qi for qi in range(arena.nQ) if qi not in used and arena.answered(uid, qi, ctx)]
        if not cand:
            cand = [qi for qi in range(arena.nQ) if qi not in used]
        # cap (labelled ceiling; approximate): most-answerable answered candidates
        if len(cand) > self.M:
            cand = sorted(cand, key=lambda qi: -p_ans_pop(arena, arena.Q[qi][0],
                          arena._answer_raw(uid, qi, ctx)[3]))[:self.M]
        tl = []; nl = []
        for qi in cand:
            tk = arena.tokens_for(uid, qi, ctx)
            tl.append(tokens + tk)
            nv = natives + ([int(arena.Q[qi][1])] if arena.is_liked_item(uid, qi, ctx) else [])
            nl.append(nv)
        Z = arena.belief_z_batch(tl, nl)
        nd = ndcg_at_k_batch(arena.FR, Z, [held] * len(tl), [prof] * len(tl), self.K)
        best = int(np.nanargmax([v if v is not None else -1 for v in nd]))
        return cand[best]


# ============================================================ B: ask-the-gradient
class AskGradient(Policy):
    """Direction g = d/dz of a smooth top-K ranking surrogate J(z)=tau*logsumexp(score/tau) through
    the FROZEN decoder: dJ/dz = sum_i softmax(score/tau)_i * W_i (expected top-item direction).
    score(q) = E_outcomes[ Delta-z(q,outcome) ] . g, expectation over the population answerer with BOTH
    tokens per outcome. snap = argmax within predicted-answerable territory (p_ans_pop > floor)."""
    name = "B_gradient"
    def __init__(self, tau=0.5, floor=0.35, M=200):
        self.tau = tau; self.floor = floor; self.M = M
    def start(self, arena, recs, ctxs):
        self.W = arena.FR.W.numpy().astype(np.float64)     # (ni,d)
    def _direction(self, arena, z):
        s = arena.FR.decode_np(z[None, :])[0]
        s = s - s.max()
        w = np.exp(s / self.tau); w /= w.sum()
        # top-mass truncation for speed: keep top 2000 items
        idx = np.argpartition(-w, 2000)[:2000]
        g = (w[idx][:, None] * self.W[idx]).sum(0)
        return g / (np.linalg.norm(g) + 1e-9)

    def pick(self, arena, i, uid, ctx, asked, ansf, used, tokens, natives, z_cur, t):
        g = self._direction(arena, z_cur)
        s, mu, sd = blind_scores(arena, z_cur)
        znorm = float(np.linalg.norm(z_cur) + 1e-9)
        # candidate subset by BLIND answerability (belief + channel prior)
        cand = top_answerable(arena, z_cur, znorm, used, self.M)
        # expected Delta-z per candidate: answered outcome, BOTH tokens, value FORECAST from belief
        tl = []; meta = []
        for qi in cand:
            ch, key = arena.Q[qi]
            pa = blind_pans(arena, z_cur, znorm, qi)
            if pa < self.floor:
                continue
            emb = arena.q_emb(qi)
            v = blind_value(arena, s, mu, sd, qi)
            toks = tokens + [(ch, KIND_IMPL, LVL_KW, 0.0, FID_DATA, 0.0, emb),
                             (ch, KIND_EXPL, LVL_ROUGH, 0.0, FID_EASE, float(v), emb)]
            tl.append(toks); meta.append((qi, pa))
        if not tl:
            return cand[0] if cand else None
        Z = arena.belief_z_batch(tl, [list(natives)] * len(tl))
        dz = Z - z_cur[None, :]
        proj = dz @ g
        return max(((qi, pa * float(pj)) for (qi, pa), pj in zip(meta, proj)),
                   key=lambda kv: kv[1])[0]


# ============================================================ C: CAT-router (Fisher-info proxy)
class CATRouter(Policy):
    """IRT-style max-information: pick the question whose answer is most informative about the user's
    latent z, weighted by answerability. Fisher-info proxy = p_ans * Var_outcome[Delta-z] magnitude
    (a question that (a) is likely answered and (b) whose outcome most MOVES the belief reduces
    posterior variance most). Pragmatic latent-Fisher on the fold's z."""
    name = "C_cat"
    def __init__(self, floor=0.30, M=200):
        self.floor = floor; self.M = M
    def start(self, arena, recs, ctxs): pass
    def pick(self, arena, i, uid, ctx, asked, ansf, used, tokens, natives, z_cur, t):
        s, mu, sd = blind_scores(arena, z_cur)
        znorm = float(np.linalg.norm(z_cur) + 1e-9)
        cand = top_answerable(arena, z_cur, znorm, used, self.M)
        # Fisher-info proxy: p_ans * ||E[Delta-z]||^2 (answerable AND belief-moving), value from belief.
        tl = []; meta = []
        for qi in cand:
            ch, key = arena.Q[qi]
            pa = blind_pans(arena, z_cur, znorm, qi)
            if pa < self.floor:
                continue
            emb = arena.q_emb(qi)
            v = blind_value(arena, s, mu, sd, qi)
            toks = tokens + [(ch, KIND_IMPL, LVL_KW, 0.0, FID_DATA, 0.0, emb),
                             (ch, KIND_EXPL, LVL_ROUGH, 0.0, FID_EASE, float(v), emb)]
            tl.append(toks); meta.append((qi, pa))
        if not tl:
            return cand[0] if cand else None
        Z = arena.belief_z_batch(tl, [list(natives)] * len(tl))
        dz = Z - z_cur[None, :]
        info = {qi: pa * float((dz[r] ** 2).sum()) for r, (qi, pa) in enumerate(meta)}
        return max(info, key=info.get)


# ============================================================ D: Golbandi ternary tree
class GolbandiTree(Policy):
    """Golbandi-2011-class conditional tree grown on TRAIN synthetic users: at each node pick the
    question that best splits users into {refuse, dislike, like} branches by resulting mean NDCG@K;
    recurse to max_depth with min_users. Answer POLARITY branching (Golbandi note). After the tree
    depth is exhausted, each user continues with the GLOBAL b2 order (skipping used) as a fixed tail --
    the tree is the ADAPTIVE part; the tail is the class-matched static continuation (documented; keeps
    construction tractable vs per-leaf greedy tails)."""
    name = "D_golbandi"

    def __init__(self, tree, b2_tail):
        self.tree = tree; self.tail = list(b2_tail)
    def start(self, arena, recs, ctxs): pass
    def pick(self, arena, i, uid, ctx, asked, ansf, used, tokens, natives, z_cur, t):
        node = self.tree
        while isinstance(node, dict) and node.get("q") is not None:
            qi = node["q"]
            if qi not in used:
                return qi
            br = _branch_static(arena, uid, qi, ctx)     # already asked -> descend by realized polarity
            node = node.get(br) or node.get("like")
        # leaf -> fixed b2 continuation, then any unused
        for qi in self.tail:
            if qi not in used:
                return qi
        for qi in range(arena.nQ):
            if qi not in used:
                return qi
        return None


def build_golbandi(arena, train_recs, b2_seq, max_depth=6, min_users=25, K=PRIMARY_K,
                   cand_cap=250, verbose=True):
    """Grow the ADAPTIVE tree (NDCG-split, polarity-branch) on TRAIN users; tail = global b2 order.
    Returns (tree, b2_seq). Cached."""
    path = f"{CACHE}/golbandi_d{max_depth}_n{len(train_recs)}.json"
    if os.path.exists(path):
        blob = json.load(open(path))
        if verbose:
            print(f"[D] loaded cached Golbandi tree from {path}", flush=True)
        return blob["tree"], b2_seq
    t0 = time.time()
    ctxs = {r["u"]: arena.user_ctx(r) for r in train_recs}
    held = {r["u"]: r["held"] for r in train_recs}
    prof = {r["u"]: set(r["known"].keys()) for r in train_recs}
    ans_rate = np.zeros(arena.nQ)
    sub = train_recs[:min(200, len(train_recs))]
    for qi in range(arena.nQ):
        ans_rate[qi] = sum(1 for r in sub if arena.answered(r["u"], qi, ctxs[r["u"]]))
    cand_pool = [int(q) for q in np.argsort(-ans_rate)[:cand_cap]]

    def user_tokens(uid, path_q):
        toks = []; nat = []; ctx = ctxs[uid]
        for qi in path_q:
            if arena.answered(uid, qi, ctx):
                toks += arena.tokens_for(uid, qi, ctx)
                if arena.is_liked_item(uid, qi, ctx):
                    nat.append(int(arena.Q[qi][1]))
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
                groups[_branch_static(arena, u, int(qi), ctxs[u])].append(u)
            if max(len(groups["dislike"]), len(groups["like"])) < min_users:
                continue
            tl = []; nl = []; order = []
            for u in uids:
                tk = arena.tokens_for(u, int(qi), ctxs[u])
                tl.append(base_tok[u][0] + tk)
                nl.append(base_tok[u][1] + ([int(arena.Q[qi][1])]
                          if arena.is_liked_item(u, int(qi), ctxs[u]) else []))
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
            print(f"[D] depth {depth} split on {arena.q_names[best_q][:34]:34s} "
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
        print(f"[D] Golbandi tree grown (depth<={max_depth}) [{time.time()-t0:.0f}s]", flush=True)
    return tree, b2_seq


def _branch_static(arena, uid, qi, ctx):
    lvl, val, fid, surp = arena._answer_raw(uid, qi, ctx)
    if lvl == LVL_NEG:
        return "refuse"
    if np.isfinite(val) and val < 0:
        return "dislike"
    return "like"


# ============================================================ b2-anchored variant wrapper
# ============================================================ A: learned scorer v2 (GBM)
def _feat(arena, qi, z_cur, s, mu, sd, znorm, n_ans, n_ref, t, Tmax, horizon):
    """BLIND feature row for (belief-state, candidate): channel one-hot, blind answerability, belief-
    region alignment, |belief| norm, blind value forecast, #answered/#refused so far, turn, horizon.
    No true surprise / true value -> deployment-safe (no profile peeking)."""
    ch, key = arena.Q[qi]
    pa = blind_pans(arena, z_cur, znorm, qi)
    emb = arena.q_emb(qi).astype(np.float64)
    align = float(emb @ z_cur) / (znorm * (np.linalg.norm(emb) + 1e-9)) if znorm > 1e-9 else 0.0
    vmag = abs(blind_value(arena, s, mu, sd, qi))
    return [1.0 if ch == TYPE_ITEM else 0.0, 1.0 if ch == TYPE_CONCEPT else 0.0,
            1.0 if ch == TYPE_ATTR else 0.0, 1.0 if ch == TYPE_ENTITY else 0.0,
            pa, align, znorm, vmag, float(n_ans), float(n_ref), t / Tmax, float(horizon)]


FEAT_NAMES = ["is_item", "is_concept", "is_attr", "is_entity", "p_ans_blind", "align", "znorm",
              "vmag_blind", "n_ans", "n_ref", "turn_frac", "horizon"]


def build_scorerA(arena, train_recs, n_samples=6000, M2=15, Tmax=24, K=PRIMARY_K, verbose=True):
    """Outcome-labelled candidate scorer: features (belief-state + candidate) -> realized 1-step AND
    2-step NDCG@K gains from simulated interviews on TRAIN users. HistGBM. Cached."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    import joblib
    path = f"{CACHE}/scorerA_n{len(train_recs)}.joblib"
    if os.path.exists(path):
        if verbose:
            print(f"[A] loaded cached GBM scorer from {path}", flush=True)
        return joblib.load(path)
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    ctxs = {r["u"]: arena.user_ctx(r) for r in train_recs}
    X, Y = [], []
    for si in range(n_samples):
        r = train_recs[rng.integers(len(train_recs))]; uid = r["u"]; ctx = ctxs[uid]
        held = r["held"]; prof = set(r["known"].keys())
        if not held:
            continue
        # random prefix (random answerable-ish questions, length 0..8)
        plen = int(rng.integers(0, 9)); used = set(); toks = []; nat = []; n_ans = 0; n_ref = 0
        allq = rng.permutation(arena.nQ)
        pi = 0
        while len(used) < plen and pi < len(allq):
            qi = int(allq[pi]); pi += 1; used.add(qi)
            if arena.answered(uid, qi, ctx):
                toks += arena.tokens_for(uid, qi, ctx); n_ans += 1
                if arena.is_liked_item(uid, qi, ctx):
                    nat.append(int(arena.Q[qi][1]))
            else:
                n_ref += 1
        z_cur = arena.belief_z_batch([toks], [nat])[0]
        s, mu, sd = blind_scores(arena, z_cur); znorm = float(np.linalg.norm(z_cur) + 1e-9)
        base = ndcg_at_k(arena.FR, z_cur, held, prof, K); base = 0.0 if base is None else base
        # candidate to score: subset by BLIND answerability (matches deploy-time ScorerA ranking)
        cpool = top_answerable(arena, z_cur, znorm, used, 80)
        qi = int(cpool[rng.integers(len(cpool))])
        tk = arena.tokens_for(uid, qi, ctx)                        # REALIZED token (label target, world)
        nv = nat + ([int(arena.Q[qi][1])] if arena.is_liked_item(uid, qi, ctx) else [])
        z1 = arena.belief_z_batch([toks + tk], [nv])[0]
        n1 = ndcg_at_k(arena.FR, z1, held, prof, K); n1 = base if n1 is None else n1
        g1 = n1 - base
        X.append(_feat(arena, qi, z_cur, s, mu, sd, znorm, n_ans, n_ref, len(used), Tmax, 1)); Y.append(g1)
        # 2-step realized gain: greedy best REALIZED follow among M2 blind-answerable candidates
        used2 = used | {qi}
        c2 = top_answerable(arena, z_cur, znorm, used2, M2)
        tl = []; nl = []
        for q2 in c2:
            tk2 = arena.tokens_for(uid, q2, ctx)
            tl.append(toks + tk + tk2)
            nl.append(nv + ([int(arena.Q[q2][1])] if arena.is_liked_item(uid, q2, ctx) else []))
        if tl:
            Z2 = arena.belief_z_batch(tl, nl)
            nd2 = ndcg_at_k_batch(arena.FR, Z2, [held] * len(tl), [prof] * len(tl), K)
            g2 = max((v if v is not None else base) for v in nd2) - base
        else:
            g2 = g1
        X.append(_feat(arena, qi, z_cur, s, mu, sd, znorm, n_ans, n_ref, len(used), Tmax, 2)); Y.append(g2)
        if verbose and (si + 1) % 1000 == 0:
            print(f"    [A] labels {si+1}/{n_samples} [{time.time()-t0:.0f}s]", flush=True)
    X = np.array(X); Y = np.array(Y)
    gbm = HistGradientBoostingRegressor(max_iter=300, max_depth=4, learning_rate=0.06,
                                        min_samples_leaf=40, random_state=SEED)
    gbm.fit(X, Y)
    joblib.dump(gbm, path)
    assert os.path.exists(path)
    if verbose:
        print(f"[A] GBM trained on {len(X)} rows (train R^2 {gbm.score(X, Y):.3f}) -> {path} "
              f"[{time.time()-t0:.0f}s]", flush=True)
    return gbm


class ScorerA(Policy):
    """Deployment: per turn, score a candidate subset with the GBM (value-to-go, horizon=2) and pick
    argmax. No folding at deploy -- features only. The CASPER-R lineage at population scale."""
    name = "A_scorer"
    def __init__(self, gbm, M=200, Tmax=24):
        self.gbm = gbm; self.M = M; self.Tmax = Tmax
    def start(self, arena, recs, ctxs): pass
    def pick(self, arena, i, uid, ctx, asked, ansf, used, tokens, natives, z_cur, t):
        n_ans = int(sum(ansf)); n_ref = int(len(ansf) - n_ans)
        s, mu, sd = blind_scores(arena, z_cur); znorm = float(np.linalg.norm(z_cur) + 1e-9)
        cand = top_answerable(arena, z_cur, znorm, used, self.M)
        X = np.array([_feat(arena, qi, z_cur, s, mu, sd, znorm, n_ans, n_ref, t, self.Tmax, 2)
                      for qi in cand])
        pred = self.gbm.predict(X)
        return cand[int(np.argmax(pred))]


# ============================================================ b2-anchored variant wrapper
class B2Anchored(Policy):
    """Follow b2's static order for the first TAU turns, then hand over to `inner` policy. TAU=Tmax
    reproduces b2 EXACTLY (tie by construction); TAU=0 is the pure policy. TAU selected on DEV-VAL."""
    def __init__(self, name, b2_seq, inner, tau):
        self.name = name; self.b2 = list(b2_seq); self.inner = inner; self.tau = tau
    def start(self, arena, recs, ctxs):
        self.inner.start(arena, recs, ctxs)
    def pick(self, arena, i, uid, ctx, asked, ansf, used, tokens, natives, z_cur, t):
        if t <= self.tau:
            for qi in self.b2:
                if qi not in used:
                    return qi
            return None
        return self.inner.pick(arena, i, uid, ctx, asked, ansf, used, tokens, natives, z_cur, t)
