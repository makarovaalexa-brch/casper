"""arena_core.py -- THE POLICY ARENA world (per DESIGN_SHEET_POLICY_ARENA.md, signed 2026-07-09).

The fair fight: does ANY deployable adaptive interview policy beat the strongest fair static on the
clean apparatus (v2.1 answerer, picked fold-v3 as belief, full question universe, E1-E7 discipline)?

WORLD (this module):
  - v2.1 answerer served LAZILY over POPULATION trU users (the 173 study/eval users NEVER touched).
    Machinery = i25_fold_v3_sampler.V3Sampler (knowledge model + real-U-EASE value model). NO LLM.
  - belief = the PICKED fold-v3 (Deep-Sets, .cache/i25_fold_v3_best.pt) -- see FOLD PICK in ARENA_BUILD.
  - question universe = FULL sampler vocabulary: all concepts + all attributes(decade/genre) + all
    IMDb entities + a FIXED popular-item bank. NO top-N pools, NO coverage sampling. Deterministic
    tie-break by question index (never alphabetical).
  - LENIENT regime: a question is ANSWERED iff the knowledge draw != no_clue (level in {rough,know_well});
    no_clue = REFUSAL = consumed turn, belief UNCHANGED, user never dropped (E1). The fold-v3 implicit-
    NEGATIVE token is DROPPED in the arena to honour the sheet's "belief unchanged on refusal" rule.
  - DETERMINISTIC per-user trait + per-(user,question) answer seed; answers memoized to disk (dense).

E-LEDGER (STATE_2026-07-08.md, applied verbatim):
  E1 same users every arm; refusal = no-op consumed turn, user kept.
  E2 baselines reproduced/constructed on the SAME synthetic world as policies (no test-fit leak).
  E3 privileged arms LABELLED, never cited as results (true-table router, clairvoyant).
  E4 paired per-user bootstrap CIs on every delta; MDE printed next to every CI.
  E5 pre-registered verdicts printed BEFORE results; no post-hoc metric selection.
  E6 sanity gates (item-static must be weak; cold orientation row).
  E7 SYMMETRY: every arm's training data / construction cohort / beliefs / fold listed side by side.

NO LLM calls. ASCII only. Deterministic seeds. Checkpoint + verify writes.
"""
import os, sys, json, time, hashlib
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "instrument2"))
import warnings
warnings.filterwarnings("ignore")

import i25_lib as L
import i25_fold_v3 as FV3
import i25_fold_v3_sampler as SP
from i25_fold_v3_sampler import (V3Sampler, TYPE_ITEM, TYPE_CONCEPT, TYPE_ATTR, TYPE_ENTITY,
                                  LVL_NEG, LVL_ROUGH, LVL_KW, KIND_IMPL, KIND_EXPL,
                                  FID_DATA, FID_EASE, FID_LLM, CENTERED_FOLD)

FOLD_CKPT = ".cache/i25_fold_v3_best.pt"                 # the PICKED fold-v3 (Deep-Sets winner)
CACHE_DIR = ".cache/arena"
CH_NAME = {TYPE_ITEM: "item", TYPE_CONCEPT: "concept", TYPE_ATTR: "attr", TYPE_ENTITY: "entity"}
CH_GROUP = {TYPE_ITEM: "item", TYPE_CONCEPT: "concept", TYPE_ATTR: "attribute", TYPE_ENTITY: "attribute"}
SEED = 123


# ============================================================ NDCG@K (generalised from i25_lib @10)
def ndcg_at_k(FR, z, tlike, profset, K, tail=False):
    """Exact NDCG@K over the FULL catalogue (profile items masked out). tlike = held-out likes."""
    S = FR.decode_np(z[None, :])[0]
    s = S.copy(); s[list(profset)] = -1e9
    if tail:
        s[FR.headmask] = -1e9
        rel = set(t for t in tlike if not FR.headmask[t])
    else:
        rel = set(tlike)
    if not rel:
        return None
    kk = min(K, len(s))
    top = np.argpartition(-s, kk - 1)[:kk]
    top = top[np.argsort(-s[top])]
    W = 1.0 / np.log2(np.arange(2, kk + 2))
    dcg = sum(W[p] for p, t in enumerate(top) if int(t) in rel)
    idcg = W[:min(kk, len(rel))].sum() + 1e-12
    return dcg / idcg


def ndcg_at_k_batch(FR, Z, held_list, prof_list, K, tail=False):
    """Vectorised NDCG@K for a batch of belief vectors Z (B,d). Returns list of float|None."""
    Sm = FR.decode_np(Z)                                     # (B, ni)
    out = []
    W = 1.0 / np.log2(np.arange(2, K + 2))
    for b in range(Sm.shape[0]):
        s = Sm[b].copy(); s[list(prof_list[b])] = -1e9
        if tail:
            s[FR.headmask] = -1e9
            rel = set(t for t in held_list[b] if not FR.headmask[t])
        else:
            rel = set(held_list[b])
        if not rel:
            out.append(None); continue
        top = np.argpartition(-s, K - 1)[:K]
        top = top[np.argsort(-s[top])]
        dcg = sum(W[p] for p, t in enumerate(top) if int(t) in rel)
        idcg = W[:min(K, len(rel))].sum() + 1e-12
        out.append(dcg / idcg)
    return out


# ============================================================ the arena
class Arena:
    def __init__(self, n_item_universe=800, verbose=True):
        t0 = time.time()
        os.makedirs(CACHE_DIR, exist_ok=True)
        self.D = L.G.load_data()
        self.FR = L.Frozen(self.D)
        self.S = V3Sampler(self.D, self.FR)
        self.model = FV3.FoldV3()
        blob = torch.load(FOLD_CKPT, map_location="cpu")
        self.model.load_state_dict(blob["model"]); self.model.eval()
        self.fold_state = blob.get("state", {})
        self._build_universe(n_item_universe)
        self.answer_cache = {}                               # (uid, qidx) -> (level, expl_val|nan, fid)
        self._emb_cache = {}                                 # qidx -> region emb (np.float32)
        self.build_emb_matrix()                              # (nQ,d) for vectorised blind answerability
        if verbose:
            print(f"[arena] loaded: fold-v3 clean_frac={self.fold_state.get('clean_frac')} "
                  f"val={self.fold_state.get('best_val')}; universe nQ={self.nQ} "
                  f"({self.n_concept} concept / {self.n_attr} attr / {self.n_entity} entity / "
                  f"{self.n_item} item)  [{time.time()-t0:.0f}s]", flush=True)

    # ---- universe: full sampler vocabulary + fixed popular-item bank -------------------------------
    def _build_universe(self, n_item):
        S = self.S
        Q = []
        for c in range(S.n_concept):
            Q.append((TYPE_CONCEPT, int(c)))
        for ak in S.attr_keys:
            Q.append((TYPE_ATTR, ak))
        for eid in S.entity_ids:
            Q.append((TYPE_ENTITY, eid))
        # FIXED popular-item bank (deterministic; tie-break by item id, NOT alphabetical)
        cnt = S.cnt
        order = sorted(range(len(cnt)), key=lambda j: (-cnt[j], j))
        top_items = order[:n_item]
        for j in top_items:
            Q.append((TYPE_ITEM, int(j)))
        self.Q = Q
        self.nQ = len(Q)
        self.n_concept = S.n_concept
        self.n_attr = len(S.attr_keys)
        self.n_entity = len(S.entity_ids)
        self.n_item = len(top_items)
        self.q_channel = np.array([q[0] for q in Q], dtype=np.int64)
        # per-question name (for transcripts)
        self.q_names = [self._qname(q) for q in Q]
        # index by channel (for candidate enumeration)
        self.idx_by_channel = {t: np.where(self.q_channel == t)[0] for t in
                               (TYPE_ITEM, TYPE_CONCEPT, TYPE_ATTR, TYPE_ENTITY)}

    def _qname(self, q):
        ch, key = q
        D = self.D
        if ch == TYPE_ITEM:
            try:
                return f"item[{D['title'][int(key)]}]"
            except Exception:
                return f"item[{int(key)}]"
        if ch == TYPE_CONCEPT:
            return f"concept[{int(key)}]"
        if ch == TYPE_ATTR:
            kind, val = key
            if kind == "dec":
                return f"attr[decade {int(val)}s]"
            gi = int(val)
            gname = L.GENRES[gi] if gi < len(L.GENRES) else str(gi)
            return f"attr[genre {gname}]"
        # entity
        e = self.S.entities.get(key, {})
        return f"entity[{e.get('name', key)} ({e.get('etype','?')})]"

    def build_emb_matrix(self):
        """Precompute (nQ, d) region-embedding matrix + per-question channel base-answerability logit,
        so blind answerability over ALL candidates is one matmul per turn (not a 1530-python-loop)."""
        if hasattr(self, "Qemb"):
            return
        import i25_fold_v3_sampler as _SP
        d = self.FR.W.shape[1]
        E = np.zeros((self.nQ, d), np.float32)
        base_logit = np.zeros(self.nQ, np.float32)
        for qi in range(self.nQ):
            E[qi] = self.q_emb(qi)
            base_logit[qi] = _SP._logit(_SP.BASE_ANS[self.Q[qi][0]])
        self.Qemb = E
        self.Qemb_norm = np.linalg.norm(E, axis=1) + 1e-9
        self.Q_base_logit = base_logit

    def blind_pans_all(self, z, znorm):
        """Vectorised blind answerability over every question: channel base + belief-alignment bump."""
        if znorm <= 1e-9:
            return 1.0 / (1.0 + np.exp(-self.Q_base_logit))
        align = (self.Qemb @ z) / (znorm * self.Qemb_norm)
        return 1.0 / (1.0 + np.exp(-(self.Q_base_logit + 1.5 * align)))

    def region_members(self, qidx):
        """Item ids belonging to a question's region (for blind value forecasting from the belief)."""
        ch, key = self.Q[qidx]
        S = self.S
        if ch == TYPE_ITEM:
            return np.array([int(key)], np.int64)
        if ch == TYPE_CONCEPT:
            return S.concept_members[int(key)]
        if ch == TYPE_ATTR:
            return S.attr_members[key]
        return S.entities[key]["members"]

    def q_emb(self, qidx):
        if qidx not in self._emb_cache:
            ch, key = self.Q[qidx]
            self._emb_cache[qidx] = np.asarray(self.S.region_emb(ch, key), np.float32)
        return self._emb_cache[qidx]

    # ---- deterministic per-user trait ------------------------------------------------------------
    def user_trait(self, uid):
        rng = np.random.default_rng((int(uid) * 2654435761) & 0xFFFFFFFF ^ SEED)
        return self.S.user_traits(rng)

    # ---- per-user precompute (known/cr/mu/trait) -------------------------------------------------
    def user_ctx(self, rec):
        known = rec["known"]
        r = np.array(list(known.values()), float)
        mu = float(r.mean())
        cr = {int(j): float(known[j] - mu) for j in known}
        return dict(known=known, mu=mu, cr=cr, trait=self.user_trait(rec["u"]))

    # ---- the answerer: deterministic answer to question qidx for user (memoized) ------------------
    def _answer_raw(self, uid, qidx, ctx):
        keyc = (int(uid), int(qidx))
        if keyc in self.answer_cache:
            return self.answer_cache[keyc]
        ch, key = self.Q[qidx]
        seed = int(hashlib.sha1(f"{uid}|{ch}|{key}".encode()).hexdigest()[:12], 16)
        rng = np.random.default_rng(seed)
        known = ctx["known"]; cr = ctx["cr"]; mu = ctx["mu"]; trait = ctx["trait"]
        surprise = self.S._surprise_val(set(int(j) for j in known), len(known), ch, key)
        level = self.S.draw_knowledge(surprise, ch, trait, rng)
        if level == LVL_NEG:
            rec = (LVL_NEG, float("nan"), -1, float(surprise))
        else:
            val, fid = self.S.region_value(known, cr, mu, ch, key, level, rng)
            if val is None:
                # answerable at knowledge level but no value producible -> still an implicit answer
                rec = (level, float("nan"), -1, float(surprise))
            else:
                rec = (level, float(val), int(fid), float(surprise))
        self.answer_cache[keyc] = rec
        return rec

    def answered(self, uid, qidx, ctx):
        """True iff the user answers (lenient regime: level != no_clue)."""
        return self._answer_raw(uid, qidx, ctx)[0] != LVL_NEG

    def tokens_for(self, uid, qidx, ctx):
        """Return the fold-v3 tokens contributed by asking qidx (empty on refusal)."""
        level, val, fid, surp = self._answer_raw(uid, qidx, ctx)
        if level == LVL_NEG:
            return []
        ch, key = self.Q[qidx]
        emb = self.q_emb(qidx)
        toks = [(ch, KIND_IMPL, level, surp, FID_DATA, 0.0, emb)]
        if np.isfinite(val):
            toks.append((ch, KIND_EXPL, LVL_ROUGH, 0.0, int(fid), float(val), emb))
        return toks

    def is_liked_item(self, uid, qidx, ctx):
        ch, key = self.Q[qidx]
        return ch == TYPE_ITEM and int(key) in ctx["known"] and ctx["known"][int(key)] >= 4

    # ---- belief: fold a set of asked questions' tokens -> z (single user) -------------------------
    def belief_z(self, uid, asked_qidx, ctx):
        toks = []
        native = []
        for qi in asked_qidx:
            toks += self.tokens_for(uid, qi, ctx)
            if self.is_liked_item(uid, qi, ctx):
                native.append(int(self.Q[qi][1]))
        return FV3.fold_np(self.FR, self.model, toks, native)

    def belief_z_batch(self, tok_lists, native_lists):
        return FV3.fold_batch(self.FR, self.model, tok_lists, native_lists)

    # ---- memoization to disk (dense answer table for a fixed user cohort) -------------------------
    def prefill_answers(self, recs, tag, verbose=True):
        """Densely compute + cache every (user,question) answer for a cohort; persist to disk."""
        path = f"{CACHE_DIR}/answers_{tag}.npz"
        if os.path.exists(path):
            blob = np.load(path, allow_pickle=True)
            uids = blob["uids"].tolist()
            lvl = blob["level"]; val = blob["val"]; fid = blob["fid"]; surp = blob["surp"]
            for a, uid in enumerate(uids):
                for qi in range(self.nQ):
                    self.answer_cache[(int(uid), qi)] = (int(lvl[a, qi]), float(val[a, qi]),
                                                         int(fid[a, qi]), float(surp[a, qi]))
            if verbose:
                print(f"[arena] loaded memoized answers {tag}: {len(uids)} users x {self.nQ} q", flush=True)
            return
        t0 = time.time()
        n = len(recs)
        lvl = np.zeros((n, self.nQ), np.int8)
        val = np.full((n, self.nQ), np.nan, np.float32)
        fid = np.full((n, self.nQ), -1, np.int8)
        surp = np.zeros((n, self.nQ), np.float32)
        uids = []
        for a, rec in enumerate(recs):
            ctx = self.user_ctx(rec); uid = rec["u"]; uids.append(int(uid))
            for qi in range(self.nQ):
                lv, v, f, s = self._answer_raw(uid, qi, ctx)
                lvl[a, qi] = lv; val[a, qi] = v; fid[a, qi] = f; surp[a, qi] = s
            if verbose and (a + 1) % 100 == 0:
                print(f"    [prefill {tag}] {a+1}/{n} users [{time.time()-t0:.0f}s]", flush=True)
        np.savez_compressed(path, uids=np.array(uids), level=lvl, val=val, fid=fid, surp=surp)
        assert os.path.exists(path)
        if verbose:
            print(f"[arena] memoized answers {tag}: {n} users x {self.nQ} q -> {path} "
                  f"[{time.time()-t0:.0f}s]", flush=True)


# ============================================================ user cohorts (TRAIN/DEV-VAL/DEV-TEST)
def make_cohorts(arena, n_train=2000, n_devval=500, n_devtest=1000, seed=SEED):
    """Disjoint SYNTHETIC user sets. Firewall: population trU users only (173 never touched)."""
    D = arena.D
    total = n_train + n_devval + n_devtest
    prof = L.load_train_profiles(D, total + 2000, seed=seed)    # over-sample; some dropped by split
    keys = list(prof.keys())
    rng = np.random.default_rng(seed); rng.shuffle(keys)
    recs = FV3.prep_users({u: prof[u] for u in keys}, np.random.default_rng(seed + 1), arena.S)
    # deterministic order by user id for reproducibility, then slice
    recs.sort(key=lambda r: r["u"])
    rng2 = np.random.default_rng(seed + 5); rng2.shuffle(recs)
    tr = recs[:n_train]
    dv = recs[n_train:n_train + n_devval]
    dt = recs[n_train + n_devval:n_train + n_devval + n_devtest]
    return dict(train=tr, devval=dv, devtest=dt)


# ============================================================ paired bootstrap + MDE
def paired_ci(deltas, n_boot=5000, seed=0):
    d = np.asarray([x for x in deltas if x is not None and np.isfinite(x)], float)
    if len(d) < 2:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), n=len(d), mde=float("nan"))
    rng = np.random.default_rng(seed)
    bs = np.empty(n_boot)
    n = len(d)
    for b in range(n_boot):
        bs[b] = d[rng.integers(0, n, n)].mean()
    lo, hi = np.percentile(bs, [2.5, 97.5])
    # MDE ~ 2.8 * SE (approx 80% power, alpha .05, two-sided) using bootstrap SE
    se = bs.std()
    return dict(mean=float(d.mean()), lo=float(lo), hi=float(hi), n=n, se=float(se),
                mde=float(2.8 * se))


if __name__ == "__main__":
    # smoke: build world, run a tiny random interview, print sanity
    ar = Arena(n_item_universe=800)
    coh = make_cohorts(ar, n_train=50, n_devval=20, n_devtest=40)
    print(f"[smoke] cohorts train={len(coh['train'])} devval={len(coh['devval'])} "
          f"devtest={len(coh['devtest'])}", flush=True)
    rec = coh["devtest"][0]; ctx = ar.user_ctx(rec)
    rng = np.random.default_rng(0)
    asked = []
    ans = 0
    for t in range(24):
        qi = int(rng.integers(ar.nQ))
        asked.append(qi)
        if ar.answered(rec["u"], qi, ctx):
            ans += 1
    z0 = ar.belief_z(rec["u"], [], ctx)
    zf = ar.belief_z(rec["u"], asked, ctx)
    n0 = ndcg_at_k(ar.FR, z0, rec["held"], set(rec["known"].keys()), 50)
    nf = ndcg_at_k(ar.FR, zf, rec["held"], set(rec["known"].keys()), 50)
    print(f"[smoke] user {rec['u']}: {ans}/24 answered; NDCG@50 cold {n0:.4f} -> after {nf:.4f}", flush=True)
    print("[smoke] channel counts in universe:",
          {CH_NAME[t]: int((ar.q_channel == t).sum()) for t in CH_NAME}, flush=True)
