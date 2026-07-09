"""i25_fold_v3_sampler.py -- the FOLD-V3 two-channel reveal sampler (per DESIGN_SHEET_FOLD_V3.md).

Generates v2.1-answerer-distributed answer sets from POPULATION trU user profiles (firewall: never
touches the 173 study / eval users). Each answered question emits up to TWO tokens (D1 separate):

  IMPLICIT token: (channel, entity, knowledge-level in {rough, know_well}) + a SURPRISE feature --
    consumption/knowledge as first-class taste evidence, NO value. The SURPRISE feature (author
    amendment) = a log-LIFT: engagement with region E relative to what the user's overall volume
    predicts (see surprise() below). NO-CLUE on an asked question emits an implicit-NEGATIVE token.
  EXPLICIT token: (channel, entity, 4-level value, fidelity-class in {data, ease, llm-style}) -- as
    v2, value from real (rated) OR EASE-backbone inference; fidelity feature learned.

Channels/type ids: 0=item, 1=concept, 2=attribute(decade/genre), 3=entity(IMDb dir/act/comp/writ/franch).
Entity embeddings: prominence(popularity)-weighted member-bag aggregate encoded by frozen RecVAE
(FR._bag_emb), + a per-type flag carried in the type one-hot.

KNOWLEDGE MODEL (the "v2.1 answerer" on population users, documented + grounded):
  P(knows-of E)  = sigmoid( logit(base_ans[ch]) + K_ANS*clip(surprise) + trait_ans_u[ch] )
  P(know_well|ans)= sigmoid( logit(base_kw[ch])  + K_KW *clip(surprise) + trait_kw_u[ch] )
  base rates from the battery A1/A3 (item famous .99/.70; concept .74/.52; attr .82/.36; entity .55/.36).
  trait_*_u ~ N(0, sigma_new[ch]) on the logit scale = the EQUATED (flutter-free) corrected trait dial
  from .cache/dans/sigma_v21.json (concept .199, entity .301, item .621). Answerability is COUPLED to
  consumption via the surprise term (K_ANS,K_KW>0): a user who consumed a region answers, and answers
  know_well when they consumed a lot -- so the implicit token is genuine taste evidence.

VALUE MODEL (real union EASE): rated item -> real centered rating (fid=data, zero noise). Inferred
  concept/attr/entity aggregate over the user's REVEALED members = real centered ratings, binned 4-level;
  unrated-item base = EASE per-item mean (ease_v21.npz mu). Fidelity/noise tied to knowledge level:
  know_well -> fid=ease (sigma 0.25*SIGMA_STAR), rough -> fid=llm-style (sigma SIGMA_STAR) = the noisy
  deployment regime the fold must be robust to (G3/G5). NO LLM calls.

NO LLM calls. Canonical scripts/caches untouched. Deterministic (seeds threaded from caller).
"""
import os, json, collections
import numpy as np

# ---- fold value scale (shared with main) --------------------------------------------------------
CENTERED_FOLD = {"hated": -1.0, "meh": -1.0 / 3.0, "liked": 1.0 / 3.0, "loved": 1.0}
SIGMA_STAR = 0.70                    # study-design fidelity sigma (stars)
CONCEPT_THR = 0.5                    # relevance threshold defining concept MEMBERSHIP (for surprise)

# token field ids
KIND_IMPL, KIND_EXPL = 0, 1
LVL_ROUGH, LVL_KW, LVL_NEG = 0, 1, 2
FID_DATA, FID_EASE, FID_LLM = 0, 1, 2
TYPE_ITEM, TYPE_CONCEPT, TYPE_ATTR, TYPE_ENTITY = 0, 1, 2, 3
NTYPE = 4

# knowledge base rates per channel (battery A1/A3; documented in module header)
BASE_ANS = {TYPE_ITEM: 0.95, TYPE_CONCEPT: 0.74, TYPE_ATTR: 0.82, TYPE_ENTITY: 0.55}
BASE_KW = {TYPE_ITEM: 0.700, TYPE_CONCEPT: 0.521, TYPE_ATTR: 0.363, TYPE_ENTITY: 0.363}
K_ANS, K_KW = 0.9, 0.5               # coupling of answerability to consumption (surprise), documented
SURPRISE_CLIP = 3.0

DANS = ".cache/dans"
EASE_CACHE = f"{DANS}/ease_v21.npz"
SIGMA_V21 = f"{DANS}/sigma_v21.json"
ATTR_BATTERY = ".cache/instrument2/attr_battery_500.json"

ENTITY_TYPES = ("director", "actor", "composer", "writer", "franchise")


def bin_star(r):
    if r >= 4.25:
        return CENTERED_FOLD["loved"]
    if r >= 3.25:
        return CENTERED_FOLD["liked"]
    if r >= 2.25:
        return CENTERED_FOLD["meh"]
    return CENTERED_FOLD["hated"]


def _logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return float(np.log(p / (1 - p)))


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


class V3Sampler:
    """Holds the region vocabulary (concepts/decades/genres/IMDb-entities), their member sets +
    popularity mass (for SURPRISE), the EQUATED per-channel trait sigmas, and EASE item means."""

    def __init__(self, D, FR):
        self.D = D
        self.FR = FR
        self.ni = int(D["ni"])
        self.cnt = np.asarray(D["cnt"], np.float64)
        self.totalpop = float(self.cnt.sum())
        self.item_tag = D["concepts"]["item_tag"]
        import i25_lib as L
        self.L = L

        # ---- corrected (equated) per-channel trait sigmas (logit scale) ----
        sg = json.load(open(SIGMA_V21))
        self.trait_sigma = {
            TYPE_ITEM: float(sg["item"]["new"]),
            TYPE_CONCEPT: float(sg["concept"]["new"]),
            TYPE_ATTR: float(sg["entity"]["new"]),   # attributes share the entity channel dial
            TYPE_ENTITY: float(sg["entity"]["new"]),
        }

        # ---- EASE per-item mean (real union EASE: inferred item base value) ----
        ed = np.load(EASE_CACHE, allow_pickle=False)
        universe = ed["universe"].astype(np.int64)
        self.ease_mu_item = np.full(self.ni, np.nan, np.float64)
        self.ease_mu_item[universe] = ed["mu"].astype(np.float64)
        self.glob_mu = float(np.nanmean(ed["mu"]))
        self.universe_set = set(int(j) for j in universe)

        # ---- CONCEPT regions ----
        self.concept_members = {}   # ctag -> np.array member ids
        self.concept_set = {}
        self.concept_popmass = {}
        nc = self.item_tag.shape[1]
        for c in range(nc):
            mem = np.where(self.item_tag[:, c] > CONCEPT_THR)[0].astype(np.int64)
            self.concept_members[c] = mem
            self.concept_set[c] = set(int(j) for j in mem)
            self.concept_popmass[c] = float(self.cnt[mem].sum())
        self.n_concept = nc

        # ---- ATTRIBUTE regions: decades + genres ----
        yrs = np.array([L._year(D, j) for j in range(self.ni)])
        self.decades = [d for d in range(1920, 2030, 10)
                        if int(((yrs >= d) & (yrs < d + 10)).sum()) >= 30]
        self.attr_members = {}      # akey -> np.array
        self.attr_set = {}
        self.attr_popmass = {}
        for d in self.decades:
            mem = np.where((yrs >= d) & (yrs < d + 10))[0].astype(np.int64)
            k = ("dec", d)
            self.attr_members[k] = mem; self.attr_set[k] = set(int(j) for j in mem)
            self.attr_popmass[k] = float(self.cnt[mem].sum())
        ng = D["Gmat"].shape[1]
        for g in range(ng):
            mem = np.where(D["Gmat"][:, g] > 0)[0].astype(np.int64)
            k = ("gen", int(g))
            self.attr_members[k] = mem; self.attr_set[k] = set(int(j) for j in mem)
            self.attr_popmass[k] = float(self.cnt[mem].sum())
        self.attr_keys = list(self.attr_members.keys())

        # ---- IMDb ENTITY regions (director/actor/composer/writer/franchise) ----
        self.entities = {}          # entity_id -> dict(members, set, popmass, etype, emb cache slot)
        self.entity_ids = []
        bat = json.load(open(ATTR_BATTERY))
        for e in bat["entities"]:
            mem = np.array([int(j) for j in e["member_dense_ids"] if 0 <= int(j) < self.ni], np.int64)
            if len(mem) == 0:
                continue
            eid = e["entity_id"]
            self.entities[eid] = dict(members=mem, mset=set(int(j) for j in mem),
                                      popmass=float(self.cnt[mem].sum() + len(mem)),
                                      etype=e["type"], name=e.get("name", eid))
            self.entity_ids.append(eid)
        self._ent_emb_cache = {}

    # ============================================================ SURPRISE feature (author amendment)
    def surprise(self, known_set, V, region_type, region_key):
        """log-LIFT of engagement with region E vs the user's volume-predicted expectation.

        n_E   = # of the user's revealed items that are MEMBERS of E.
        p_E   = popularity mass of E / total popularity  (base membership rate for a pop-weighted watcher).
        E[n_E]= V * p_E   (volume-predicted expectation).
        surprise = log( (n_E + 0.5) / (E[n_E] + 0.5) ).

        Selective user (all of X, small V): n_E ~ V, E[n_E] small -> LARGE positive surprise.
        Prolific user (all of X + everything, large V): n_E same, V large -> surprise ~ 0. (G2b.)
        """
        if region_type == TYPE_CONCEPT:
            mset = self.concept_set[region_key]; pm = self.concept_popmass[region_key]
        elif region_type == TYPE_ATTR:
            mset = self.attr_set[region_key]; pm = self.attr_popmass[region_key]
        elif region_type == TYPE_ENTITY:
            mset = self.entities[region_key]["mset"]; pm = self.entities[region_key]["popmass"]
        else:  # item: E = {the single item}
            j = int(region_key)
            n_E = 1.0 if j in known_set else 0.0
            pm = float(self.cnt[j] + 1.0)
            p_E = pm / self.totalpop
            exp = max(V, 1) * p_E
            return float(np.log((n_E + 0.5) / (exp + 0.5)))
        n_E = float(sum(1 for j in known_set if j in mset))
        p_E = pm / self.totalpop
        exp = max(V, 1) * p_E
        return float(np.log((n_E + 0.5) / (exp + 0.5))), n_E

    def _n_E(self, known_set, region_type, region_key):
        if region_type == TYPE_CONCEPT:
            mset = self.concept_set[region_key]
        elif region_type == TYPE_ATTR:
            mset = self.attr_set[region_key]
        elif region_type == TYPE_ENTITY:
            mset = self.entities[region_key]["mset"]
        else:
            return 1.0 if int(region_key) in known_set else 0.0
        return float(sum(1 for j in known_set if j in mset))

    def _surprise_val(self, known_set, V, region_type, region_key):
        s = self.surprise(known_set, V, region_type, region_key)
        return s[0] if isinstance(s, tuple) else s

    # ============================================================ region embeddings
    def region_emb(self, region_type, region_key):
        if region_type == TYPE_ITEM:
            return self.FR.Wn[int(region_key)].numpy().astype(np.float32)
        if region_type == TYPE_CONCEPT:
            return self.FR.concept_emb(int(region_key))
        if region_type == TYPE_ATTR:
            return self.FR.attr_emb(region_key)
        # IMDb entity: prominence(pop+1)-weighted member-bag encoded by frozen RecVAE
        if region_key not in self._ent_emb_cache:
            mem = self.entities[region_key]["members"]
            w = np.zeros(self.ni, np.float64)
            w[mem] = self.cnt[mem] + 1.0
            self._ent_emb_cache[region_key] = self.FR._bag_emb(w)
        return self._ent_emb_cache[region_key]

    # ============================================================ per-user trait draw (corrected dial)
    def user_traits(self, rng):
        """Per-user knowledge trait (logit-scale) per channel, N(0, sigma_new[ch]) -- the EQUATED dial.
        One shared draw per channel used for both the answerability and know_well shifts."""
        return {t: float(rng.normal(0.0, self.trait_sigma[t])) for t in
                (TYPE_ITEM, TYPE_CONCEPT, TYPE_ATTR, TYPE_ENTITY)}

    # ============================================================ knowledge draw for one asked region
    def draw_knowledge(self, surprise, region_type, trait, rng):
        s = float(np.clip(surprise, -SURPRISE_CLIP, SURPRISE_CLIP))
        p_ans = _sig(_logit(BASE_ANS[region_type]) + K_ANS * s + trait[region_type])
        if rng.random() >= p_ans:
            return LVL_NEG  # no_clue -> implicit-negative
        p_kw = _sig(_logit(BASE_KW[region_type]) + K_KW * s + trait[region_type])
        return LVL_KW if rng.random() < p_kw else LVL_ROUGH

    # ============================================================ value for one region (real U EASE)
    def region_value(self, known, cr, mu, region_type, region_key, level, rng):
        """Return (value, fid) for the EXPLICIT token, or (None,None) if no value producible.
        cr = {j: centered real rating}, mu = revealed-mean (star centering)."""
        noise = 0.0
        if level == LVL_KW:
            fid = FID_EASE; noise = 0.25 * SIGMA_STAR
        else:
            fid = FID_LLM; noise = SIGMA_STAR
        if region_type == TYPE_ITEM:
            j = int(region_key)
            if j in cr:                                   # real rating -> data token (zero noise)
                return float(cr[j]), FID_DATA
            base = self.ease_mu_item[j]                   # EASE per-item mean (real U EASE)
            if not np.isfinite(base):
                base = self.glob_mu
            v = float(np.clip(base + rng.normal(0, noise), 0.5, 5.0))
            return bin_star(v), fid
        # concept/attr/entity aggregate over the user's REVEALED members (real ratings)
        if region_type == TYPE_CONCEPT:
            members = self.concept_members[region_key]
            rel = self.item_tag[np.fromiter(cr.keys(), np.int64), region_key]
            kk = np.fromiter(cr.keys(), np.int64)
            w = rel; crv = np.array([cr[int(j)] for j in kk])
            m = float(w.sum())
            if m < 1e-6:
                return None, None
            agg = float((w * crv).sum() / m)
        else:
            if region_type == TYPE_ATTR:
                mset = self.attr_set[region_key]
            else:
                mset = self.entities[region_key]["mset"]
            vals = [cr[int(j)] for j in cr if int(j) in mset]
            if not vals:
                return None, None
            agg = float(np.mean(vals))
        v = bin_star(agg + mu + rng.normal(0, noise))
        return v, fid

    # ============================================================ emit tokens for one asked region
    def emit(self, known, cr, mu, region_type, region_key, trait, rng, force_level=None):
        """Return list of tokens for ONE asked question (up to 2: implicit + explicit)."""
        known_set = set(int(j) for j in known)
        V = len(known)
        surprise = self._surprise_val(known_set, V, region_type, region_key)
        emb = self.region_emb(region_type, region_key)
        level = force_level if force_level is not None else \
            self.draw_knowledge(surprise, region_type, trait, rng)
        toks = []
        if level == LVL_NEG:
            # no_clue on an ASKED question -> implicit-NEGATIVE token only (weak "not in world")
            toks.append((region_type, KIND_IMPL, LVL_NEG, surprise, FID_DATA, 0.0, emb))
            return toks
        # IMPLICIT token (knowledge as taste evidence, NO value)
        toks.append((region_type, KIND_IMPL, level, surprise, FID_DATA, 0.0, emb))
        # EXPLICIT token (value from real U EASE)
        val, fid = self.region_value(known, cr, mu, region_type, region_key, level, rng)
        if val is not None:
            toks.append((region_type, KIND_EXPL, LVL_ROUGH, 0.0, fid, float(val), emb))
        return toks

    # ============================================================ per-user region cache (speed)
    def make_cache(self, known):
        """Precompute the user's engaged regions ONCE (reused across epochs/questions) so interview
        sampling is O(1) per question instead of scanning the whole entity/attr vocabulary."""
        ks = list(int(j) for j in known)
        kset = set(ks)
        it_arr = np.array(ks, np.int64)
        mass = self.item_tag[it_arr].sum(0) if len(it_arr) else np.zeros(self.n_concept)
        concept_top = [int(c) for c in np.argsort(-mass)[:20] if mass[c] > 1e-6]
        attr_hit = [ak for ak in self.attr_keys if not self.attr_set[ak].isdisjoint(kset)]
        ent_hit = [eid for eid, e in self.entities.items() if not e["mset"].isdisjoint(kset)]
        liked = [j for j in ks if known[j] >= 4]
        return dict(ks=ks, kset=kset, mass=mass, concept_top=concept_top, attr_hit=attr_hit,
                    ent_hit=ent_hit, liked=liked)

    # ============================================================ build a full reveal (curriculum)
    def build_reveal(self, known, mode, budget, rng, trait=None, cache=None, max_items=None):
        """mode='clean' (full profile) or 'interview' (partial, `budget` questions, mixed channels).
        max_items caps the per-item clean tokens for training throughput (None = true full profile,
        used by the G4 gate). Returns (tokens, native_liked_ids). `known` = {item_id: rating}."""
        if trait is None:
            trait = self.user_traits(rng)
        if cache is None:
            cache = self.make_cache(known)
        ks = cache["ks"]
        r = np.array([known[j] for j in ks], float)
        mu = float(r.mean())
        cr = {int(j): float(known[j] - mu) for j in ks}
        native = cache["liked"]
        if mode == "clean":
            return self._build_clean(known, cr, mu, native, trait, rng, cache, max_items)
        return self._build_interview(known, cr, mu, native, budget, trait, rng, cache)

    def _build_clean(self, known, cr, mu, native, trait, rng, cache, max_items=None):
        """Clean full profile: every rated item -> explicit(data) + implicit(know_well) token; plus
        top concept/attr/entity aggregates. This is what closes G4 (matches native RecVAE)."""
        toks = []
        ks = cache["ks"]; kset = cache["kset"]
        if max_items is not None and len(ks) > max_items:
            idx = rng.choice(len(ks), size=max_items, replace=False)
            ks = [ks[i] for i in idx]
        for j in ks:
            j = int(j)
            surprise = self._surprise_val(kset, len(ks), TYPE_ITEM, j)
            emb = self.FR.Wn[j].numpy().astype(np.float32)
            toks.append((TYPE_ITEM, KIND_EXPL, LVL_ROUGH, 0.0, FID_DATA, float(cr[j]), emb))
            toks.append((TYPE_ITEM, KIND_IMPL, LVL_KW, surprise, FID_DATA, 0.0, emb))
        # top concepts by revealed relevance mass
        for c in cache["concept_top"][:6]:
            toks += self.emit(known, cr, mu, TYPE_CONCEPT, int(c), trait, rng, force_level=LVL_KW)
        # top attributes by revealed member count
        agg = [(ak, sum(1 for j in kset if j in self.attr_set[ak])) for ak in cache["attr_hit"]]
        for ak, _ in sorted(agg, key=lambda kv: -kv[1])[:6]:
            toks += self.emit(known, cr, mu, TYPE_ATTR, ak, trait, rng, force_level=LVL_KW)
        # top IMDb entities the user engaged
        ent_hit = [(sum(1 for j in kset if j in self.entities[eid]["mset"]), eid)
                   for eid in cache["ent_hit"]]
        for _, eid in sorted(ent_hit, reverse=True)[:4]:
            toks += self.emit(known, cr, mu, TYPE_ENTITY, eid, trait, rng, force_level=LVL_KW)
        rng.shuffle(toks)
        return toks, native

    def _pick_region(self, ch, rng, on_profile, cache):
        """Pick a region (entity) in channel `ch` using the precomputed user cache. on_profile ->
        bias toward regions the user engaged; else a random region (may yield no_clue)."""
        if ch == TYPE_ITEM:
            ks = cache["ks"]
            if on_profile and ks:
                return int(ks[rng.integers(len(ks))])
            return int(rng.integers(self.ni))            # random (likely unrated) item
        if ch == TYPE_CONCEPT:
            top = cache["concept_top"]
            if on_profile and top:
                return int(top[rng.integers(len(top))])
            return int(rng.integers(self.n_concept))
        if ch == TYPE_ATTR:
            hit = cache["attr_hit"]
            if on_profile and hit:
                return hit[rng.integers(len(hit))]
            return self.attr_keys[rng.integers(len(self.attr_keys))]
        hit = cache["ent_hit"]
        if on_profile and hit:
            return hit[rng.integers(len(hit))]
        return self.entity_ids[rng.integers(len(self.entity_ids))]

    SLOT_MIX = [(TYPE_ITEM, 0.30), (TYPE_CONCEPT, 0.30), (TYPE_ATTR, 0.20), (TYPE_ENTITY, 0.20)]
    _SLOT_CH = [c for c, _ in SLOT_MIX]
    _SLOT_CDF = np.cumsum([p for _, p in SLOT_MIX])

    def _build_interview(self, known, cr, mu, native, budget, trait, rng, cache):
        toks = []
        asked = set()
        for _ in range(budget):
            ch = self._SLOT_CH[int(np.searchsorted(self._SLOT_CDF, rng.random()))]
            on_profile = rng.random() < 0.7
            key = self._pick_region(ch, rng, on_profile, cache)
            sig = (ch, key if not isinstance(key, np.integer) else int(key))
            if sig in asked:
                continue
            asked.add(sig)
            toks += self.emit(known, cr, mu, ch, key, trait, rng)
        # native prior = revealed LIKED items among item tokens actually asked
        item_asked = [int(k) for (c, k) in asked if c == TYPE_ITEM and int(k) in known and known[int(k)] >= 4]
        rng.shuffle(toks)
        return toks, item_asked
