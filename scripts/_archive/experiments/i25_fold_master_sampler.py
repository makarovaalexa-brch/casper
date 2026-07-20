"""i25_fold_master_sampler.py -- the LOCKED FOLD-MASTER reveal sampler (per FOLD_MASTER.md sec E).

AMENDS the proven V3Sampler (i25_fold_v3_sampler) with the six locked design changes:
  1. NO SURPRISE (F4). Tokens carry only leak-free fields: IMPLICIT (channel, member-bag emb,
     knowledge-level {rough,know_well}); EXPLICIT (channel, member-bag emb, value 4-level, fidelity).
  2. ITEM-HOLE (R6): native_z = LIKED revealed items only; every consumed item ALSO emits
     implicit(know_well)+explicit(data) TOKENS so a consumed-not-liked item carries GoT in z-space
     through rho (never into native_z).
  3. Token tuple = (typ, kind, lvl, fid, val, emb, dkey, turn). dkey identifies (channel,entity) for
     DISTINCT-set dedup; turn = turn index for the deterministic collision tie-break. No surprise field.
  4. Any-strategy curriculum: per-interview on_profile rate drawn from a strategy mixture (on/mostly/
     mixed/explore/random) -> natural refusals; LOG-UNIFORM lengths 1..len(known); 30% clean; NO caps.
  5. Refusal (no_clue) emits NO token (informs answerability, not taste; excluded from the fold).

Reuses V3Sampler's PROVEN data-side machinery unchanged: region vocab, member-bag embeddings
(genome x popularity only), value model (real U EASE), knowledge model (v2.1 answerer draw). NO LLM.
"""
import numpy as np
from i25_fold_v3_sampler import (V3Sampler, KIND_IMPL, KIND_EXPL, LVL_ROUGH, LVL_KW, LVL_NEG,
                                 FID_DATA, FID_EASE, FID_LLM, TYPE_ITEM, TYPE_CONCEPT, TYPE_ATTR,
                                 TYPE_ENTITY, NTYPE, CENTERED_FOLD, bin_star)

# strategy mixture = per-interview on_profile probability (R2 any-strategy; low p -> natural refusals)
STRATEGIES = {"onprofile": 1.0, "mostly_on": 0.8, "mixed": 0.5, "explore": 0.2, "random": 0.0}
_STRAT_NAMES = list(STRATEGIES.keys())


def _sanitize_emb(emb):
    """DATA HYGIENE (F6, q718 class): zero any non-finite embedding row."""
    e = np.asarray(emb, np.float32)
    if not np.all(np.isfinite(e)):
        e = np.zeros_like(e)
    return e


def dedup_key(ch, key):
    if ch == TYPE_ITEM:
        return f"I:{int(key)}"
    if ch == TYPE_CONCEPT:
        return f"C:{int(key)}"
    if ch == TYPE_ATTR:
        return f"A:{key[0]}:{key[1]}"
    return f"E:{key}"


class MasterSampler(V3Sampler):
    # -------- one region -> up to 2 tokens (implicit + explicit), NO surprise, with dkey+turn --------
    def emit_region(self, known, cr, mu, ch, key, level, turn, rng):
        if level == LVL_NEG:
            return []                                     # refusal: no token (answerability, not taste)
        emb = _sanitize_emb(self.region_emb(ch, key))
        dk = dedup_key(ch, key)
        toks = [(ch, KIND_IMPL, level, FID_DATA, 0.0, emb, dk, turn)]
        val, fid = self.region_value(known, cr, mu, ch, key, level, rng)
        if val is not None:
            toks.append((ch, KIND_EXPL, LVL_ROUGH, fid, float(val), emb, dk, turn))
        return toks

    def _item_tokens(self, cr, j, turn):
        """A consumed item: implicit(know_well) + explicit(data, real centered rating). Item-hole
        carrier -- these tokens fire for EVERY consumed item (liked or not); only LIKED enter native_z."""
        emb = _sanitize_emb(self.FR.Wn[int(j)].numpy().astype(np.float32))
        dk = dedup_key(TYPE_ITEM, j)
        return [(TYPE_ITEM, KIND_IMPL, LVL_KW, FID_DATA, 0.0, emb, dk, turn),
                (TYPE_ITEM, KIND_EXPL, LVL_ROUGH, FID_DATA, float(cr[int(j)]), emb, dk, turn)]

    # ------------------------------------------------------------------ full reveal (curriculum)
    def build_reveal(self, known, mode, budget, rng, trait=None, cache=None, max_items=None,
                     strategy=None):
        if trait is None:
            trait = self.user_traits(rng)
        if cache is None:
            cache = self.make_cache(known)
        ks = cache["ks"]
        r = np.array([known[j] for j in ks], float)
        mu = float(r.mean())
        cr = {int(j): float(known[j] - mu) for j in ks}
        if mode == "clean":
            return self._build_clean(known, cr, mu, trait, rng, cache)
        return self._build_interview(known, cr, mu, budget, trait, rng, cache, strategy)

    def _build_clean(self, known, cr, mu, trait, rng, cache):
        """Clean full profile: EVERY rated item -> impl(kw)+expl(data) token (NO cap, R10); liked ->
        native_z; plus every engaged concept/attr/entity aggregate. This closes G-clean."""
        toks = []
        ks = cache["ks"]; kset = cache["kset"]
        native = []
        turn = 0
        for j in ks:
            j = int(j)
            toks += self._item_tokens(cr, j, turn); turn += 1
            if known[j] >= 4:
                native.append(j)
        for c in cache["concept_top"]:
            toks += self.emit_region(known, cr, mu, TYPE_CONCEPT, int(c), LVL_KW, turn, rng); turn += 1
        agg = [(ak, sum(1 for j in kset if j in self.attr_set[ak])) for ak in cache["attr_hit"]]
        for ak, _ in sorted(agg, key=lambda kv: -kv[1]):
            toks += self.emit_region(known, cr, mu, TYPE_ATTR, ak, LVL_KW, turn, rng); turn += 1
        ent_hit = [(sum(1 for j in kset if j in self.entities[eid]["mset"]), eid)
                   for eid in cache["ent_hit"]]
        for _, eid in sorted(ent_hit, reverse=True):
            toks += self.emit_region(known, cr, mu, TYPE_ENTITY, eid, LVL_KW, turn, rng); turn += 1
        rng.shuffle(toks)
        return toks, native

    def _build_interview(self, known, cr, mu, budget, trait, rng, cache, strategy=None):
        if strategy is None:
            strategy = _STRAT_NAMES[rng.integers(len(_STRAT_NAMES))]
        p_on = STRATEGIES[strategy]
        toks = []
        asked = set()
        native = []
        turn = 0
        for _ in range(int(budget)):
            ch = self._SLOT_CH[int(np.searchsorted(self._SLOT_CDF, rng.random()))]
            on_profile = rng.random() < p_on
            key = self._pick_region(ch, rng, on_profile, cache)
            kk = key if not isinstance(key, np.integer) else int(key)
            sig = (ch, kk)
            if sig in asked:
                continue
            asked.add(sig)
            if ch == TYPE_ITEM:
                j = int(kk)
                if j in known:                                   # consumed -> know_well item tokens
                    toks += self._item_tokens(cr, j, turn)
                    if known[j] >= 4:
                        native.append(j)
                else:                                            # unseen item: may know via fame/EASE
                    lvl = self.draw_knowledge(0.0, TYPE_ITEM, trait, rng)
                    toks += self.emit_region(known, cr, mu, TYPE_ITEM, j, lvl, turn, rng)
            else:
                lvl = self.draw_knowledge(0.0, ch, trait, rng)
                toks += self.emit_region(known, cr, mu, ch, kk, lvl, turn, rng)
            turn += 1
        rng.shuffle(toks)
        return toks, native


def loguniform_budget(nmax, rng):
    """LOG-UNIFORM length 1..nmax (fills the 24->full gap; no caps)."""
    nmax = max(int(nmax), 1)
    if nmax == 1:
        return 1
    b = int(round(float(np.exp(rng.uniform(np.log(1.0), np.log(nmax))))))
    return int(min(max(b, 1), nmax))
