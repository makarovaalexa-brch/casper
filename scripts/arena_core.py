"""arena_core.py -- THE POLICY ARENA world, GATED v2.1 answerer (per DESIGN_SHEET_POLICY_ARENA.md,
signed 2026-07-09; rebuilt per ARENA_CODE_AUDIT.md + BLIND_VALIDATION.md F14, 2026-07-10).

WORLD (fix #7 -- the certification-chain repair):
  - The answerer is THE FITTED, GATED v2.1 MODEL (.cache/dans/models_v21.json): per-channel ordinal
    logistics with the full feature set (co-knowledge, fame, census/buffness, era) + EQUATED
    (flutter-free) per-user trait sigma; entity per-cut random effects. Loaded and RUN, not
    approximated. Knowledge generation = dans_stages.know_probs + _sample_cat, verbatim recipe of
    generate_population (deterministic per-user seed SEED*7777+uid).
  - Values = the v2.1 EASE-backbone value models (stage_value design): t(u,i) = real rating if rated
    else per-user EASE prediction (ease_v21.npz B); item X=[t-3.5, fame, genre_align, cmean];
    concept/entity X=[pop-weighted member-t aggregate-3.5, has_rated_member, log1p(n_rated), cmean];
    ordinal 4-level sample. Rated bank items pass through: knowledge=know_well, value=real rating
    (fid=data). NO LLM calls.
  - UNIVERSE (fix #5, per the signed sheet): ALL judged questions = 1,128 concepts + 500 attributes
    (IMDb entities) + 800 bank items = 2,428. No pools, no coverage sampling; deterministic layout
    [concept | entity | item] (= generate_population's layout).
  - Belief = FOLD-V3.1 (.cache/i25_fold_v31_best.pt; NOCLUE_ABLATION verdict): the anti-surprise
    retrain, val 0.4356 > v3 0.4311, decisive gates pass with G2b (+0.433) and G7 (+0.032) STRONGER.
    NOTE (documented deviation): the fold was TRAINED on the hand-parameterized sampler world; it is
    deployed FIXED on the gated world (mild distribution shift, same token vocabulary).
  - LENIENT regime + v3.1 REFUSAL RULE: know>=1 = answered; know=0 = REFUSAL = consumed turn, user
    never dropped (E1) -- and the no-clue token (with ANTI-SURPRISE) IS FOLDED (v3.1-fold+anti beats
    v3-skip +0.0098 CI excl 0; the earlier skip rule + inert-no-clue caveat are SUPERSEDED).
  - CACHES (fix #4): per-cohort answer tables keyed by a config sha (seed, all cohort sizes, nq,
    models file sha) AND per-user known-set hashes verified on load; mismatch = recompute.

E-LEDGER (STATE_2026-07-08.md, verbatim): E1 same users, refusal=no-op; E2 constructions on TRAIN
synthetic world; E3 privileged arms labelled; E4 paired bootstrap + MDE; E5 verdicts pre-registered;
E6 sanity rows; E7 symmetry table printed. ASCII. Deterministic. Firewall: population trU users only
(study ids excluded); the 173/300 are NEVER touched.
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
import i25_fold_v31 as FV31
import dans_build as DB
import dans_stages as DS
from i25_fold_v3_sampler import (TYPE_ITEM, TYPE_CONCEPT, TYPE_ATTR, TYPE_ENTITY,
                                  LVL_ROUGH, LVL_KW, LVL_NEG, KIND_IMPL, KIND_EXPL,
                                  FID_DATA, FID_EASE, FID_LLM, CENTERED_FOLD)

FOLD_CKPT = ".cache/i25_fold_v31_best.pt"      # FOLD-V3.1 (no-clue anti-surprise; NOCLUE_ABLATION)
MODELS21 = ".cache/dans/models_v21.json"
EASE_V21 = ".cache/dans/ease_v21.npz"
EQUATE_V21 = ".cache/dans/equate_v21.json"
CACHE_DIR = ".cache/arena"
SEED = 123
CH_NAME = {TYPE_ITEM: "item", TYPE_CONCEPT: "concept", TYPE_ENTITY: "entity"}
# centered fold value per 4-level value index (hated, meh, liked, loved)
VBIN_CENTERED = np.array([CENTERED_FOLD["hated"], CENTERED_FOLD["meh"],
                          CENTERED_FOLD["liked"], CENTERED_FOLD["loved"]], np.float64)
HEADLINE_MARGIN = {"concept": 1, "entity": 1, "item": 2}


# ============================================================ NDCG@K (unchanged, audited-sound)
def ndcg_at_k(FR, z, tlike, profset, K, tail=False):
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
    Sm = FR.decode_np(Z)
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


# ============================================================ the gated arena
class Arena:
    def __init__(self, verbose=True):
        t0 = time.time()
        os.makedirs(CACHE_DIR, exist_ok=True)
        self.uni = DB.Universe(verbose=False)
        self.D = self.uni.D
        self.FR = L.Frozen(self.D)
        self.model = FV31.FoldV31()
        blob = torch.load(FOLD_CKPT, map_location="cpu")
        self.model.load_state_dict(blob["model"]); self.model.eval()
        self.fold_state = blob.get("state", {})
        # ---- fitted gated v2.1 models ----
        self.models = json.load(open(MODELS21))
        for grp in ("knowledge", "value"):
            for ch in self.models[grp]:
                for k in ("mu", "sd", "theta"):
                    self.models[grp][ch][k] = np.asarray(self.models[grp][ch][k], float)
        assert self.models["meta"].get("sigma_source", "").startswith("equated"), \
            "models_v21.json must carry the equated sigma_u"
        self.models_sha = DB.sha(MODELS21)
        # ---- EASE value backbone ----
        ed = np.load(EASE_V21)
        self.ease_B = ed["B"]                                  # (9352, 9352) float32
        self.ease_uni = ed["universe"].astype(np.int64)
        self.ease_mu = ed["mu"].astype(np.float64)
        self.ease_index = {int(j): c for c, j in enumerate(self.ease_uni)}
        # ---- universe layout [concept | entity | item] (generate_population layout) ----
        u = self.uni
        self.ntag, self.nent, self.nbank = u.ntag, u.nent, u.nbank
        self.off_ent = self.ntag
        self.off_item = self.ntag + self.nent
        self.nQ = self.ntag + self.nent + self.nbank
        self.q_channel = np.concatenate([
            np.full(self.ntag, TYPE_CONCEPT, np.int64),
            np.full(self.nent, TYPE_ENTITY, np.int64),
            np.full(self.nbank, TYPE_ITEM, np.int64)])
        self.q_names = ([f"concept[{t.get('tag', t['tagId'])}]" for t in u.tags] +
                        [f"entity[{e.get('name', e['entity_id'])} ({e.get('type','?')})]" for e in u.ents] +
                        [f"item[{self.D['title'][int(j)]}]" for j in u.bank])
        # popularity mass per question (for the SURPRISE token feature; same formula as fold training)
        cnt = self.D["cnt"].astype(np.float64)
        self.totalpop = float(cnt.sum())
        pm_tag = np.asarray(u.tagM.dot(cnt)).ravel()
        pm_ent = np.asarray(u.entM.dot(cnt)).ravel() + np.asarray(u.entM.sum(1)).ravel()
        pm_item = cnt[u.bank] + 1.0
        self.q_popmass = np.concatenate([pm_tag, pm_ent, pm_item])
        self._emb_cache = {}
        self.user_tables = {}                                  # uid -> dict(know, val, crval, surp)
        self.build_emb_matrix()
        if verbose:
            print(f"[arena] GATED world: fitted v2.1 models (sha {self.models_sha}) + EASE backbone "
                  f"({len(self.ease_uni)} items); universe nQ={self.nQ} "
                  f"({self.ntag} concept / {self.nent} entity / {self.nbank} item) = the signed "
                  f"sheet's 2,428; fold-v3 val={self.fold_state.get('best_val'):.4f}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    # ---- region embedding per question (fold token entity emb) --------------------------------
    def q_emb(self, qidx):
        if qidx not in self._emb_cache:
            u = self.uni
            if qidx < self.off_ent:                            # concept: pop-weighted member bag
                mem = u.tagM[qidx].indices
                w = np.zeros(u.ni, np.float64); w[mem] = self.D["cnt"][mem] + 1.0
                e = self.FR._bag_emb(w)
            elif qidx < self.off_item:                         # entity: pop-weighted member bag
                mem = u.entM[qidx - self.off_ent].indices
                w = np.zeros(u.ni, np.float64); w[mem] = self.D["cnt"][mem] + 1.0
                e = self.FR._bag_emb(w)
            else:                                              # item: decoder row
                e = self.FR.Wn[int(u.bank[qidx - self.off_item])].numpy().astype(np.float32)
            self._emb_cache[qidx] = np.asarray(e, np.float32)
        return self._emb_cache[qidx]

    def build_emb_matrix(self):
        """(nQ,d) embedding matrix for vectorised blind-alignment (built once, cached to disk)."""
        p = f"{CACHE_DIR}/qemb_{self.nQ}.npz"
        if os.path.exists(p):
            self.Qemb = np.load(p)["E"]
        else:
            t0 = time.time()
            self.Qemb = np.stack([self.q_emb(q) for q in range(self.nQ)])
            np.savez_compressed(p, E=self.Qemb)
            assert os.path.exists(p)
            print(f"[arena] built q-embedding matrix ({self.nQ}) [{time.time()-t0:.0f}s]", flush=True)
        self.Qemb_norm = np.linalg.norm(self.Qemb, axis=1) + 1e-9

    def region_members(self, qidx):
        u = self.uni
        if qidx < self.off_ent:
            return u.tagM[qidx].indices.astype(np.int64)
        if qidx < self.off_item:
            return u.entM[qidx - self.off_ent].indices.astype(np.int64)
        return np.array([int(u.bank[qidx - self.off_item])], np.int64)

    # ---- per-user GATED answer-table generation (deterministic; generate_population recipe) ----
    def _gen_user(self, uid, known):
        u = self.uni
        f = u.user_features(known)
        rng = np.random.default_rng(SEED * 7_777 + int(uid))
        Pk = DS.know_probs(u, f, self.models, rng=rng)
        know = np.concatenate([DS._sample_cat(Pk["concept"], rng),
                               DS._sample_cat(Pk["entity"], rng),
                               DS._sample_cat(Pk["item"], rng)]).astype(np.int8)
        # ---- v2.1 EASE-backbone VALUES (stage_value design; NOT the iter-2 value features) ----
        cmean = float(f["cmean"])
        rc = np.zeros(len(self.ease_mu))
        for j, r in known.items():
            c = self.ease_index.get(int(j))
            if c is not None:
                rc[c] = r - self.ease_mu[c]
        pred = np.clip(self.ease_mu + rc @ self.ease_B, 0.5, 5.0)
        t_full = np.full(u.ni, np.nan)
        t_full[self.ease_uni] = pred
        known_ind = np.zeros(u.ni, np.float64)
        for j, r in known.items():
            if 0 <= int(j) < u.ni:
                t_full[int(j)] = r
                known_ind[int(j)] = 1.0
        tm = np.isfinite(t_full).astype(np.float64)
        tv = np.where(tm > 0, t_full, 0.0)
        wt = u.pr * tm
        mv = self.models["value"]
        # concept aggregate (pop-weighted member t)
        num_c = np.asarray(u.tagM.dot(wt * tv)).ravel()
        den_c = np.asarray(u.tagM.dot(wt)).ravel()
        ctaste = np.where(den_c > 1e-9, num_c / np.maximum(den_c, 1e-9), cmean)
        nr_c = np.asarray(u.tagM.dot(known_ind)).ravel()
        Xc = np.column_stack([ctaste - 3.5, (nr_c > 0).astype(float), np.log1p(nr_c),
                              np.full(self.ntag, cmean)])
        Pv_c = DB.ord_prob(mv["concept"]["theta"], DB.zscale(Xc, mv["concept"]["mu"],
                           mv["concept"]["sd"]), 4)
        # entity aggregate
        num_e = np.asarray(u.entM.dot(wt * tv)).ravel()
        den_e = np.asarray(u.entM.dot(wt)).ravel()
        etaste = np.where(den_e > 1e-9, num_e / np.maximum(den_e, 1e-9), cmean)
        nr_e = np.asarray(u.entM.dot(known_ind)).ravel()
        Xe = np.column_stack([etaste - 3.5, (nr_e > 0).astype(float), np.log1p(nr_e),
                              np.full(self.nent, cmean)])
        Pv_e = DB.ord_prob(mv["entity"]["theta"], DB.zscale(Xe, mv["entity"]["mu"],
                           mv["entity"]["sd"]), 4)
        # item (EASE t direct)
        t_bank = t_full[u.bank]
        t_bank = np.where(np.isfinite(t_bank), t_bank, cmean)
        Xi = np.column_stack([t_bank - 3.5, f["item_val"][:, 1], f["item_val"][:, 0],
                              np.full(self.nbank, cmean)])
        Pv_i = DB.ord_prob(mv["item"]["theta"], DB.zscale(Xi, mv["item"]["mu"], mv["item"]["sd"]), 4)
        val = np.full(self.nQ, -1, np.int8)
        for off, P in ((0, Pv_c), (self.off_ent, Pv_e), (self.off_item, Pv_i)):
            seg = know[off:off + P.shape[0]]
            nz = seg > 0
            if nz.any():
                dv = DS._sample_cat(P[nz], rng)
                block = np.full(P.shape[0], -1, np.int8); block[nz] = dv
                val[off:off + P.shape[0]] = block
        # rated-item passthrough: knowledge=know_well; value = real rating (token uses cr, fid=data)
        mu_known = float(np.mean(list(known.values())))
        crval = np.full(self.nbank, np.nan, np.float32)
        rated = f["rated_flag"] > 0.5
        if rated.any():
            know[self.off_item:][rated] = 2
            for r in np.where(rated)[0]:
                star = known[int(u.bank[r])]
                val[self.off_item + r] = 3 if star >= 4.5 else 2 if star >= 3.5 else \
                    1 if star >= 2.5 else 0
                crval[r] = np.float32(star - mu_known)
        # SURPRISE per question (fold token feature; same formula the fold was trained with)
        V = max(len(known), 1)
        n_E = np.concatenate([np.asarray(u.tagM.dot(known_ind)).ravel(),
                              np.asarray(u.entM.dot(known_ind)).ravel(),
                              known_ind[u.bank]])
        p_E = self.q_popmass / self.totalpop
        surp = np.log((n_E + 0.5) / (V * p_E + 0.5)).astype(np.float32)
        return dict(know=know, val=val, crval=crval, surp=surp)

    def user_table(self, uid, known):
        if uid not in self.user_tables:
            self.user_tables[uid] = self._gen_user(uid, known)
        return self.user_tables[uid]

    # ---- world/answer interface --------------------------------------------------------------
    def user_ctx(self, rec):
        t = self.user_table(rec["u"], rec["known"])
        return dict(known=rec["known"], table=t)

    def answered(self, uid, qidx, ctx):
        return int(ctx["table"]["know"][qidx]) >= 1

    def answer_level(self, uid, qidx, ctx):
        """0=no_clue(refusal), 1=rough, 2=know_well."""
        return int(ctx["table"]["know"][qidx])

    def realized_branch(self, uid, qidx, ctx):
        """Answer-polarity branch for an ALREADY-ASKED question (observed dialogue; legal for D)."""
        t = ctx["table"]
        k = int(t["know"][qidx])
        if k == 0:
            return "refuse"
        v = int(t["val"][qidx])
        return "dislike" if 0 <= v <= 1 else "like"

    def tokens_for(self, uid, qidx, ctx):
        """FOLD-V3.1 tokens (8-field, incl ANTI-SURPRISE) for asking qidx. NEW REFUSAL RULE
        (NOCLUE_ABLATION verdict): no_clue -> the implicit-NEGATIVE token IS FOLDED, carrying
        anti-surprise = clip(-surprise, 0, 8) (expectedness of knowing); v3's skip rule and its
        inert-no-clue caveat are SUPERSEDED by v3.1's measured semantics (expected refusals pull
        away -0.059; v3.1-fold+anti beats v3-skip +0.0098 CI excl 0)."""
        t = ctx["table"]
        k = int(t["know"][qidx])
        ch = int(self.q_channel[qidx])
        emb = self.Qemb[qidx]
        surp = float(t["surp"][qidx])
        if k == 0:
            anti = float(np.clip(-surp, 0.0, 8.0))
            return [(ch, KIND_IMPL, LVL_NEG, surp, FID_DATA, 0.0, emb, anti)]
        lvl = LVL_KW if k == 2 else LVL_ROUGH
        toks = [(ch, KIND_IMPL, lvl, surp, FID_DATA, 0.0, emb, 0.0)]
        if qidx >= self.off_item and np.isfinite(t["crval"][qidx - self.off_item]):
            toks.append((ch, KIND_EXPL, LVL_ROUGH, 0.0, FID_DATA,
                         float(t["crval"][qidx - self.off_item]), emb, 0.0))
        else:
            v = int(t["val"][qidx])
            if v >= 0:
                fid = FID_EASE if k == 2 else FID_LLM
                toks.append((ch, KIND_EXPL, LVL_ROUGH, 0.0, fid, float(VBIN_CENTERED[v]), emb, 0.0))
        return toks

    def is_liked_item(self, uid, qidx, ctx):
        if qidx < self.off_item:
            return False
        j = int(self.uni.bank[qidx - self.off_item])
        return j in ctx["known"] and ctx["known"][j] >= 4

    def belief_z_batch(self, tok_lists, native_lists):
        return FV31.fold_batch(self.FR, self.model, tok_lists, native_lists)

    def belief_z(self, uid, asked_qidx, ctx):
        toks = []; native = []
        for qi in asked_qidx:
            toks += self.tokens_for(uid, qi, ctx)          # incl no-clue tokens (v3.1 rule)
            if self.is_liked_item(uid, qi, ctx) and self.answered(uid, qi, ctx):
                native.append(int(self.uni.bank[qi - self.off_item]))
        return FV31.fold_np(self.FR, self.model, toks, native)

    # ---- vectorised BLIND answerability (population per-question prior + belief alignment) ----
    def set_pop_prior(self, p0_q):
        """Per-question population answer-rate prior (computed on TRAIN cohort; shared by ALL blind
        adaptive arms -- E7 symmetric, fix for audit finding 7)."""
        p = np.clip(np.asarray(p0_q, float), 1e-3, 1 - 1e-3)
        self.p0_logit = np.log(p / (1 - p))

    def blind_pans_all(self, z, znorm):
        assert hasattr(self, "p0_logit"), "set_pop_prior must be called before blind policies run"
        if znorm <= 1e-9:
            return 1.0 / (1.0 + np.exp(-self.p0_logit))
        align = (self.Qemb @ z) / (znorm * self.Qemb_norm)
        return 1.0 / (1.0 + np.exp(-(self.p0_logit + 1.5 * align)))

    # ---- cohort-config-keyed memoization (fix #4) ----------------------------------------------
    def _cohort_cfg_sha(self, cfg):
        s = json.dumps(dict(seed=SEED, nq=self.nQ, models=self.models_sha, **cfg), sort_keys=True)
        return hashlib.sha1(s.encode()).hexdigest()[:12]

    @staticmethod
    def _known_hash(known):
        s = ",".join(str(j) for j in sorted(int(k) for k in known))
        return np.uint64(int(hashlib.sha1(s.encode()).hexdigest()[:15], 16))

    def prefill_answers(self, recs, name, cfg, verbose=True):
        """Densely generate + cache every user's full answer table; per-user known-hash verified on
        load (stale/mismatched cohort config -> regenerate; fix #4)."""
        sha_c = self._cohort_cfg_sha(cfg)
        path = f"{CACHE_DIR}/answers2_{name}_{sha_c}.npz"
        if os.path.exists(path):
            b = np.load(path)
            uids = b["uids"].tolist()
            kh = b["known_hash"]
            recmap = {r["u"]: r for r in recs}
            ok = len(uids) == len(recs)
            if ok:
                for a, uid in enumerate(uids):
                    r = recmap.get(int(uid))
                    if r is None or self._known_hash(r["known"]) != kh[a]:
                        ok = False; break
            if ok:
                for a, uid in enumerate(uids):
                    self.user_tables[int(uid)] = dict(know=b["know"][a], val=b["val"][a],
                                                      crval=b["crval"][a], surp=b["surp"][a])
                if verbose:
                    print(f"[arena] loaded verified answer tables {name} ({len(uids)} users, "
                          f"cfg {sha_c})", flush=True)
                return
            print(f"[arena] cache {path} STALE (known-hash mismatch) -> regenerating", flush=True)
        t0 = time.time()
        n = len(recs)
        know = np.zeros((n, self.nQ), np.int8)
        val = np.zeros((n, self.nQ), np.int8)
        crval = np.zeros((n, self.nbank), np.float32)
        surp = np.zeros((n, self.nQ), np.float32)
        kh = np.zeros(n, np.uint64)
        uids = []
        for a, rec in enumerate(recs):
            t = self.user_table(rec["u"], rec["known"])
            know[a] = t["know"]; val[a] = t["val"]; crval[a] = t["crval"]; surp[a] = t["surp"]
            kh[a] = self._known_hash(rec["known"]); uids.append(int(rec["u"]))
            if verbose and (a + 1) % 200 == 0:
                print(f"    [gen {name}] {a+1}/{n} users [{time.time()-t0:.0f}s]", flush=True)
        np.savez_compressed(path, uids=np.array(uids), know=know, val=val, crval=crval,
                            surp=surp, known_hash=kh)
        assert os.path.exists(path)
        if verbose:
            print(f"[arena] generated + cached answer tables {name}: {n} users x {self.nQ} q "
                  f"(cfg {sha_c}) [{time.time()-t0:.0f}s]", flush=True)


# ============================================================ cohorts (study ids EXCLUDED)
def make_cohorts(arena, n_train=1000, n_devval=80, n_devtest=160, seed=SEED):
    """Disjoint synthetic cohorts from population trU users, study ids excluded. Split convention =
    population_split (per-user rng seed*1_000_003+u; first half known); held = LIKED (r>=4) items of
    the other half. Per-user split is uid-seeded => INDEPENDENT of cohort sizes (fix #4 root cause)."""
    excl = DB.study_ids()
    total = n_train + n_devval + n_devtest
    d = np.load(DB.META)
    uu = d["uu"].astype(np.int64); ii = d["ii"].astype(np.int64); rr = d["rr"].astype(np.float32)
    order = np.argsort(uu, kind="stable")
    uu, ii, rr = uu[order], ii[order], rr[order]
    bnd = np.searchsorted(uu, np.arange(uu[-1] + 2))
    trU = d["trU"].astype(np.int64)
    cand = np.array([u for u in trU if u not in excl], np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(cand)
    out = []
    for u in cand:
        s, e = bnd[u], bnd[u + 1]
        its = ii[s:e]; rat = rr[s:e]
        if len(its) < 8:
            continue
        ru = np.random.default_rng(seed * 1_000_003 + int(u))
        perm = ru.permutation(len(its))
        half = len(its) // 2
        known = {int(its[k]): float(rat[k]) for k in perm[:half]}
        held = set(int(its[k]) for k in perm[half:] if rat[k] >= 4)
        if len(known) < 4 or not held:
            continue
        out.append(dict(u=int(u), known=known, held=held))
        if len(out) >= total:
            break
    assert len(out) >= total, f"cohort shortfall: {len(out)} < {total}"          # audit finding 11
    return dict(train=out[:n_train], devval=out[n_train:n_train + n_devval],
                devtest=out[n_train + n_devval:total],
                cfg=dict(n_train=n_train, n_devval=n_devval, n_devtest=n_devtest, split_seed=seed))


# ============================================================ paired bootstrap + MDE (E4)
def paired_ci(deltas, n_boot=5000, seed=0):
    d = np.asarray([x for x in deltas if x is not None and np.isfinite(x)], float)
    if len(d) < 2:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), n=len(d),
                    se=float("nan"), mde=float("nan"))
    rng = np.random.default_rng(seed)
    bs = np.empty(n_boot)
    n = len(d)
    for b in range(n_boot):
        bs[b] = d[rng.integers(0, n, n)].mean()
    lo, hi = np.percentile(bs, [2.5, 97.5])
    se = bs.std()
    return dict(mean=float(d.mean()), lo=float(lo), hi=float(hi), n=n, se=float(se),
                mde=float(2.8 * se))


# ============================================================ WORLD FUEL CHECK (gate stamp, fix #7)
def world_fuel_check(arena, recs, out_json=f"{CACHE_DIR}/world_fuel_check.json"):
    """Mini fuel reproduction ON THE ARENA WORLD AS BUILT: per-channel trait ICC (question-
    residualized between-user variance share) at the headline margin vs the CORRECTED targets
    (equate_v21.json icc_trait_corrected). >=400 synthetic users. The arena has no call structure,
    so flutter=0 by construction and the one-way user share IS the trait ICC. Tolerance 0.02 (the
    v2.1 G2 rule)."""
    assert len(recs) >= 400, "fuel check needs >=400 users"
    eq = json.load(open(EQUATE_V21))["decomp"]
    res = {}
    spans = {"concept": (0, arena.off_ent), "entity": (arena.off_ent, arena.off_item),
             "item": (arena.off_item, arena.nQ)}
    K = np.stack([arena.user_table(r["u"], r["known"])["know"] for r in recs])
    for ch, (lo, hi) in spans.items():
        thr = HEADLINE_MARGIN[ch]
        Y = (K[:, lo:hi] >= thr).astype(np.float64)              # (n_users, n_q)
        qbar = Y.mean(0)
        Rm = Y - qbar[None, :]                                   # question-residualized
        a, m = Rm.shape
        gm = Rm.mean()
        MS_U = m * ((Rm.mean(1) - gm) ** 2).sum() / (a - 1)
        MS_E = ((Rm - Rm.mean(1, keepdims=True)) ** 2).sum() / (a * (m - 1))
        sig2_U = max((MS_U - MS_E) / m, 0.0)
        icc = sig2_U / max(sig2_U + MS_E, 1e-12)
        tgt = eq[ch][str(thr)]["icc_trait_corrected"]
        res[ch] = dict(margin=thr, arena_trait_icc=float(icc), corrected_target=float(tgt),
                       match=bool(abs(icc - tgt) <= 0.02), base_rate=float(Y.mean()))
    res["all_match"] = all(res[ch]["match"] for ch in spans)
    res["n_users"] = len(recs)
    json.dump(res, open(out_json, "w"), indent=1)
    print("\n=== WORLD FUEL CHECK (arena world as built; trait ICC vs CORRECTED targets, tol 0.02) ===",
          flush=True)
    for ch in spans:
        r = res[ch]
        print(f"  {ch:8s} k>={r['margin']}: arena trait ICC {r['arena_trait_icc']:.4f} vs corrected "
              f"target {r['corrected_target']:.4f} -> {'MATCH' if r['match'] else 'MISS'} "
              f"(base rate {r['base_rate']:.3f})", flush=True)
    print(f"  ALL MATCH: {res['all_match']} (n={len(recs)} synthetic users; flutter=0 by "
          f"construction)", flush=True)
    return res


if __name__ == "__main__":
    ar = Arena()
    coh = make_cohorts(ar, n_train=30, n_devval=10, n_devtest=20)
    rec = coh["devtest"][0]; ctx = ar.user_ctx(rec)
    rng = np.random.default_rng(0)
    asked = [int(rng.integers(ar.nQ)) for _ in range(24)]
    ans = sum(1 for qi in asked if ar.answered(rec["u"], qi, ctx))
    z0 = ar.belief_z(rec["u"], [], ctx)
    zf = ar.belief_z(rec["u"], asked, ctx)
    n0 = ndcg_at_k(ar.FR, z0, rec["held"], set(rec["known"].keys()), 50)
    nf = ndcg_at_k(ar.FR, zf, rec["held"], set(rec["known"].keys()), 50)
    print(f"[smoke] user {rec['u']}: {ans}/24 answered; NDCG@50 cold {n0:.4f} -> {nf:.4f}", flush=True)
    t = ctx["table"]["know"]
    for ch, (lo, hi) in (("concept", (0, ar.off_ent)), ("entity", (ar.off_ent, ar.off_item)),
                         ("item", (ar.off_item, ar.nQ))):
        print(f"[smoke] {ch:8s} answer rate {float((t[lo:hi] >= 1).mean()):.3f} "
              f"know_well rate {float((t[lo:hi] >= 2).mean()):.3f}", flush=True)
