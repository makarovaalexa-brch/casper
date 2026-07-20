"""habitat_experiment.py -- THE ONE CORRECTIVE EXPERIMENT from experiments/FRESH_AUDIT.md.

Removes the three highest-ranked masks the audit found, in ONE local run (NO LLM API calls):

  FINDING 1 (pool excludes the fuel): rebuild the candidate pool HABITAT-STRATIFIED. Per channel:
    ~20 top-coverage questions + ~40 from the SCARCITY HABITAT (population answer rate 0.2..0.7),
    selected by POPULATION statistics only (no per-user peeking). Refusals now cost turns, so
    answerability discovery has real purchase for the first time.

  FINDING 2 (fuel-free sim training world): refit the VALUE model on the REAL grid, LEAVE-ONE-USER-OUT
    (construction-half users only, per estimate) instead of the population sampler; refit the KNOWLEDGE
    belief (LOUO logistic MF, Stage B machinery) on the construction half only.

  FINDING 3 (no taste-conditioned policy): a hand-built ROUTER that folds the first answers -> taste
    direction, then selects habitat questions ranked by P(answerable | events) x taste-alignment
    cos(z_t, emb) x population-value. Two variants: tie-by-construction (warm-started on the split-fair
    static) and free.

  FINDING 5 (metric looks where the fuel isn't): report ENDPOINT NDCG@10 AND @50 alongside anytime@10,
    at T=12 and T=24. PRE-REGISTERED PRIMARY = endpoint NDCG@50 at T=24 (gives adaptivity room to pay
    back exploration; anytime@10 saturates by t4-6). Chosen plainly, before results.

  FINDING 6/7 (baseline asymmetry / power): split-fair statics constructed on the SAME construction half
    as the router's beliefs (E7 arm-symmetry). Old top-coverage-pool static as a reference row. Every
    "tie" reported WITH its MDE (min detectable effect, 80% power). Paired per-user bootstrap.

TWO ARENAS (author addition 2026-07-09):
  LENIENT (--arena lenient): answered iff knowledge k>=1 (rough_idea folds a value token). The
    original audit spec; knowledge belief targets the arena's own answerability P(k>=1).
  STRICT  (--arena strict):  answered iff knowledge k>=2 (know_well). A rough_idea response is a
    HEDGE: it consumes the turn, folds NO value token, and feeds y=0 to the online knowledge
    belief (the hedge still updates the knowledge map -- the observation sub-variant is inherent
    in the event recording). Rationale (author): all measured fuel (ICC, habitat band, LOUO
    predictability) lives at know_well; the lenient construct made popular questions ~99%
    answerable (fuel-free). In the strict arena the answer rates BECOME the k>=2 rates --
    refusal-rich and user-specific everywhere. Belief targets P(know_well) exactly.
  Pool habitat band, value model, beliefs, statics all use the ARENA's own answerability (E7).

E-ledger E1-E7 honored (see experiments/HABITAT_EXPERIMENT.md symmetry table). DIRECTIONAL 173/300.
Run:  python scripts/habitat_experiment.py --do all --arena lenient|strict
"""
import os, sys, json, time, argparse, collections
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings("ignore")
import adaptivity_battery_v1 as A
import battery_stage_b as B
import i25_lib as L
import i25_fold_v2 as V2
import i25_phase4_fair as FA
import battery_stage_c as SC_C          # KmapItem

CKPT = ".cache/i25_fold_v2_best.pt"
OUT_MD = "experiments/HABITAT_EXPERIMENT.md"
OUT_JSON = "experiments/habitat_experiment.json"
CKPT_JSON = "experiments/habitat_experiment_ckpt.json"

ARENA = "lenient"               # set by configure(); "strict" => answered iff know_well (k>=2)
KTHR = 1                        # answerability knowledge threshold (1=lenient, 2=strict)


def configure(arena):
    """Set arena globals + output paths. STRICT: answered iff k>=2; hedge = turn burned, no token,
    y=0 to the knowledge belief (belief therefore targets P(know_well) -- arena-matched)."""
    global ARENA, KTHR, OUT_MD, OUT_JSON, CKPT_JSON
    ARENA = arena
    KTHR = 2 if arena == "strict" else 1
    suf = "_STRICT" if arena == "strict" else ""
    OUT_MD = f"experiments/HABITAT_EXPERIMENT{suf}.md"
    OUT_JSON = f"experiments/habitat_experiment{suf.lower()}.json"
    CKPT_JSON = f"experiments/habitat_experiment{suf.lower()}_ckpt.json"

CF = V2.CENTERED_FOLD
FID = V2.FID

# ---- pool design (audit: per channel 20 top-coverage + 40 habitat @ rate 0.2..0.7) ----
N_TOPCOV = 20
N_HABITAT = 40
HAB_LO, HAB_HI = 0.20, 0.70
MIN_ASKED = 10                  # population support: a question must be asked by >=10 users to qualify
OLD_POOL = 60                   # the old top-coverage-per-channel pool (reference static)

BUDGETS = [12, 24]
TMAX = max(BUDGETS)
KS = [10, 50]
PRIMARY_T = 24
PRIMARY_K = 50
WARMUP = 3                      # anchored router follows s-fair for the first WARMUP turns (E2 anchor)
BOOT = 5000
SEED = 0
Z_POWER = 2.801                 # z_{0.975}+z_{0.80} for 80%-power two-sided MDE


def md(txt, mode="a"):
    os.makedirs("experiments", exist_ok=True)
    open(OUT_MD, mode, encoding="utf-8").write(txt)


def paired(a, b, seed=SEED):
    """Paired per-user bootstrap. Returns delta, ci, n, means, sd(of paired diffs), MDE(80% power)."""
    pa, pb = [], []
    for x, y in zip(a, b):
        if x is None or y is None:
            continue
        if (isinstance(x, float) and np.isnan(x)) or (isinstance(y, float) and np.isnan(y)):
            continue
        pa.append(x); pb.append(y)
    if not pa:
        return dict(delta=float("nan"), ci=[float("nan")] * 2, n=0, a=float("nan"), b=float("nan"),
                    sd=float("nan"), mde=float("nan"))
    d = np.asarray(pa) - np.asarray(pb)
    rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(BOOT)])
    sd = float(np.std(d, ddof=1)) if len(d) > 1 else float("nan")
    mde = Z_POWER * sd / np.sqrt(len(d)) if len(d) > 1 else float("nan")
    return dict(delta=float(d.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                n=int(len(d)), a=float(np.mean(pa)), b=float(np.mean(pb)), sd=sd, mde=float(mde))


# =========================================================== token + batched NDCG (v2 fold, any K)
def tok_of(CANDS, rec, cid):
    m = CANDS[cid]
    return (m["typ"], int(rec["fid_arr"][cid]), m["emb"], float(rec["val_arr"][cid]))


CHUNK = 3000
def batch_ndcg_k(FR, model, users, tl, nl, idx, kk):
    out = np.empty(len(tl))
    for s in range(0, len(tl), CHUNK):
        e = min(s + CHUNK, len(tl))
        Z = V2.fold_batch_v2(FR, model, tl[s:e], nl[s:e])
        out[s:e] = FA.ndcg_batch(FR, Z, [users[idx[r]] for r in range(s, e)], kk)
    return out


def cold_k(FR, users, kk):
    Z = np.zeros((len(users), FR.W.shape[1]), np.float32)
    return FA.ndcg_batch(FR, Z, users, kk)


# =========================================================== pool construction (habitat-stratified)
def _fill_arrays(users, CANDS, cid_of):
    nc = len(CANDS)
    for rec in users:
        cm = rec["cmean"]
        ans = np.zeros(nc, bool); val = np.zeros(nc); fid = np.zeros(nc, np.int64)
        nat = [None] * nc; klev = -np.ones(nc, np.int64); src = [None] * nc
        for c in rec["concept"]:
            cd = cid_of.get(("concept", c["tagId"]))
            if cd is not None and c["k"] is not None:
                klev[cd] = c["k"]                    # recorded even when hedged/refused (trace readout)
            if cd is not None and c["k"] is not None and c["k"] >= KTHR:
                ans[cd] = True; val[cd] = CF.get((c["ans"] or {}).get("value"), 0.0)
                fid[cd] = FID["llm_kw"] if c["k"] >= 2 else FID["llm_rough"]
                klev[cd] = c["k"]; src[cd] = "concept"
        for c in rec["attribute"]:
            cd = cid_of.get(("attr", c["eid"]))
            if cd is not None and c["k"] is not None:
                klev[cd] = c["k"]
            if cd is not None and c["k"] is not None and c["k"] >= KTHR:
                ans[cd] = True; val[cd] = CF.get((c["ans"] or {}).get("value"), 0.0)
                fid[cd] = FID["llm_kw"] if c["k"] >= 2 else FID["llm_rough"]
                klev[cd] = c["k"]; src[cd] = "attr"
        for c in rec["item_llm"]:
            cd = cid_of.get(("item", c["j"]))
            if cd is not None and c["k"] is not None:
                klev[cd] = c["k"]
            if cd is not None and c["k"] is not None and c["k"] >= KTHR:
                stars = (c["ans"] or {}).get("stars")
                if stars is not None:
                    ans[cd] = True; val[cd] = (stars - cm)
                    fid[cd] = FID["llm_kw"] if c["k"] >= 2 else FID["llm_rough"]
                    klev[cd] = c["k"]; src[cd] = "llm"
                    if stars >= 4:
                        nat[cd] = c["j"]
        for c in rec["item_data"]:
            cd = cid_of.get(("item", c["j"]))
            if cd is not None:
                ans[cd] = True; val[cd] = (c["stars"] - cm); fid[cd] = FID["data"]
                klev[cd] = 2; src[cd] = "data"
                if c["stars"] >= 4:
                    nat[cd] = c["j"]
        rec["ans_arr"] = ans; rec["val_arr"] = val; rec["fid_arr"] = fid
        rec["nat_arr"] = nat; rec["klev_arr"] = klev; rec["src_arr"] = src


def _channel_stats(users):
    """Population asked/answered counts per (channel,key). Rate = answered(k>=KTHR)/asked(parsed).
    In the STRICT arena the rates ARE the know_well rates (author: where the fuel lives)."""
    asked = collections.Counter(); ans = collections.Counter()
    for rec in users:
        for c in rec["concept"]:
            if c["k"] is not None:
                asked[("concept", c["tagId"])] += 1
                if c["k"] >= KTHR:
                    ans[("concept", c["tagId"])] += 1
        for c in rec["attribute"]:
            if c["k"] is not None:
                asked[("attr", c["eid"])] += 1
                if c["k"] >= KTHR:
                    ans[("attr", c["eid"])] += 1
        for c in rec["item_llm"]:
            if c["k"] is not None:
                asked[("item", c["j"])] += 1
                if c["k"] >= KTHR:
                    ans[("item", c["j"])] += 1
    return asked, ans


def _emb_builders(D, FR, memb, battery):
    ni = int(D["ni"])

    def concept_emb_g(tagId):
        mems = [j for j in memb.get(str(tagId), []) if 0 <= j < ni]
        if not mems:
            return None
        w = np.zeros(ni, np.float64); w[mems] = D["cnt"][mems] + 1.0
        return FR._bag_emb(w)

    def attr_emb_g(eid):
        e = battery.get(eid)
        if not e:
            return None
        mems = [j for j in e.get("member_dense_ids", []) if 0 <= j < ni]
        if not mems:
            return None
        w = np.zeros(ni, np.float64); w[mems] = D["cnt"][mems] + 1.0
        return FR._bag_emb(w)

    return concept_emb_g, attr_emb_g


def build_pool(D, FR, users, memb, battery, habitat=True):
    """Habitat-stratified pool (habitat=True) or OLD top-coverage-60 pool (habitat=False).
    Returns (CANDS, meta) and fills per-user *_arr. meta carries per-question rate/asked/ans/habitat."""
    concept_emb_g, attr_emb_g = _emb_builders(D, FR, memb, battery)
    asked, ansc = _channel_stats(users)
    nU = len(users)

    def rate(key):
        a = asked.get(key, 0)
        return (ansc.get(key, 0) / a) if a else 0.0

    selected = []      # list of (channel, key, is_habitat)
    for ch in ("concept", "item", "attr"):
        keys = [k for (c, k) in asked if c == ch]
        # coverage = answered count (matches prior `cov`)
        by_cov = sorted(keys, key=lambda k: -ansc.get((ch, k), 0))
        if not habitat:
            for k in by_cov[:OLD_POOL]:
                selected.append((ch, k, False))
            continue
        topcov = by_cov[:N_TOPCOV]
        chosen = set(topcov)
        # habitat: rate in [LO,HI], asked>=MIN_ASKED, not already chosen; rank by asked-count desc
        hab = [k for k in keys if k not in chosen and asked.get((ch, k), 0) >= MIN_ASKED
               and HAB_LO <= rate((ch, k)) <= HAB_HI]
        hab.sort(key=lambda k: -asked.get((ch, k), 0))
        hab = hab[:N_HABITAT]
        for k in topcov:
            selected.append((ch, k, False))
        for k in hab:
            selected.append((ch, k, True))

    CANDS = []
    for ch, key, is_hab in selected:
        if ch == "concept":
            emb = concept_emb_g(int(key)); typ = 1
        elif ch == "item":
            emb = FR.Wn[key].numpy().astype(np.float32); typ = 0
        else:
            emb = attr_emb_g(key); typ = 2
        if emb is None:
            continue
        CANDS.append(dict(cid=len(CANDS), kind=ch, key=key, emb=np.asarray(emb, np.float32), typ=typ,
                          cov=ansc.get((ch, key), 0) / nU, asked=asked.get((ch, key), 0),
                          ans=ansc.get((ch, key), 0), rate=rate((ch, key)), habitat=bool(is_hab)))
    cid_of = {(m["kind"], m["key"]): m["cid"] for m in CANDS}
    _fill_arrays(users, CANDS, cid_of)
    return CANDS, cid_of


# =========================================================== split-fair greedy static (no cov cap)
def build_greedy(env, CANDS, pool_cids, sub_users, sub_cold, T, kk):
    """Greedy forward selection maximizing cohort-mean endpoint NDCG@kk over sub_users. Prefix-
    consistent (one T build serves all budgets). NO coverage cap (the whole point vs the old pool)."""
    FR, model = env["FR"], env["model"]
    pool = list(pool_cids)
    n = len(sub_users); sched = []
    for pos in range(T):
        cands = [c for c in pool if c not in sched]
        if not cands:
            break
        tl, nl, idx, owner = [], [], [], []
        base_ans = [[c for c in sched if rec["ans_arr"][c]] for rec in sub_users]
        for ci, c in enumerate(cands):
            for i, rec in enumerate(sub_users):
                cids = base_ans[i] + ([c] if rec["ans_arr"][c] else [])
                if not cids:
                    continue
                tl.append([tok_of(CANDS, rec, cc) for cc in cids])
                nl.append([rec["nat_arr"][cc] for cc in cids if rec["nat_arr"][cc] is not None])
                idx.append(i); owner.append(ci)
        ndcg = batch_ndcg_k(FR, model, sub_users, tl, nl, idx, kk)
        sums = np.zeros(len(cands)); got = collections.defaultdict(set)
        for r, ci in enumerate(owner):
            sums[ci] += ndcg[r]; got[ci].add(idx[r])
        best, bc = -1.0, None
        for ci, c in enumerate(cands):
            mean = (sums[ci] + sum(sub_cold[i] for i in range(n) if i not in got[ci])) / n
            if mean > best:
                best, bc = mean, c
        sched.append(bc)
    return sched


def eval_plan_metrics(env, CANDS, plans, sub_users, T):
    """plans[i] = per-turn list of (cid or None). Returns dict with per-turn @10 and @50 matrices +
    per-user refusal-rate array (fraction of the first T turns that were refusals)."""
    FR, model = env["FR"], env["model"]
    n = len(sub_users)
    cold10 = cold_k(FR, sub_users, 10); cold50 = cold_k(FR, sub_users, 50)
    pt = {10: np.empty((n, T)), 50: np.empty((n, T))}
    for t in range(T):
        tl, nl, idx = [], [], []
        for i, rec in enumerate(sub_users):
            cids = [c for c in plans[i][:t + 1] if c is not None]
            if not cids:
                continue
            tl.append([tok_of(CANDS, rec, c) for c in cids])
            nl.append([rec["nat_arr"][c] for c in cids if rec["nat_arr"][c] is not None])
            idx.append(i)
        for kk in KS:
            col = (cold10 if kk == 10 else cold50).copy()
            if tl:
                col[idx] = batch_ndcg_k(FR, model, sub_users, tl, nl, idx, kk)
            pt[kk][:, t] = col
    refusal = np.array([sum(1 for c in plans[i][:T] if c is None) / T for i in range(n)])
    return pt, refusal


def static_plans(CANDS, sched, sub_users, T):
    return [[(c if rec["ans_arr"][c] else None) for c in sched[:T]] for rec in sub_users]


# =========================================================== value model (REAL grid, construction half)
def fit_value_pop(env, CANDS, sub_users, kk=PRIMARY_K):
    """V_pop[cid] = mean single-answer NDCG@kk lift from cold over construction-half users who can
    answer it (population value on the REAL grid; firewall = eval users excluded). No per-user peek."""
    FR, model = env["FR"], env["model"]
    cold = cold_k(FR, sub_users, kk)
    tl, nl, mi, mc = [], [], [], []
    for i, rec in enumerate(sub_users):
        for cd in np.where(rec["ans_arr"])[0]:
            tl.append([tok_of(CANDS, rec, int(cd))])
            nl.append([rec["nat_arr"][int(cd)]] if rec["nat_arr"][int(cd)] is not None else [])
            mi.append(i); mc.append(int(cd))
    vals = batch_ndcg_k(FR, model, sub_users, tl, nl, mi, kk) if tl else np.array([])
    lifts = collections.defaultdict(list)
    for r in range(len(vals)):
        lifts[mc[r]].append(vals[r] - cold[mi[r]])
    V = np.zeros(len(CANDS))
    for cd in range(len(CANDS)):
        if lifts[cd]:
            V[cd] = float(np.mean(lifts[cd]))
    return V


# =========================================================== knowledge belief (LOUO MF, construction half)
def train_channel_mf(sub_users, CANDS, kind):
    """Logistic MF (Stage B) on construction-half users' answered(k>=KTHR) cells for one channel.
    Arena-matched target: lenient P(k>=1); STRICT P(know_well). Returns dict(E,b,erow)."""
    keys = sorted({m["key"] for m in CANDS if m["kind"] == kind})
    erow = {k: i for i, k in enumerate(keys)}
    kfield = {"concept": ("concept", "tagId"), "attr": ("attribute", "eid")}[kind]
    listname, idname = kfield
    pos = []
    for rec in sub_users:
        for c in rec[listname]:
            key = c[idname]
            if key in erow and c["k"] is not None and c["k"] >= KTHR:
                pos.append((rec["u"], erow[key]))
    if not pos:
        return None
    E, b = B.train_mf(pos, len(keys), B.D_MF, seed=SEED)
    return dict(E=E, b=b, erow=erow)


class Belief:
    """Online answerability belief per channel. Concept/attr via LOUO-MF; item via population kmap."""
    def __init__(self, sub_users, CANDS, KM):
        self.KM = KM
        self.mf = {}
        for kind in ("concept", "attr"):
            self.mf[kind] = train_channel_mf(sub_users, CANDS, kind)

    def p_ans(self, CANDS, cid, ev_by_kind):
        """P(answerable) for candidate cid given events grouped by kind: ev_by_kind[kind]=(keys,ys)."""
        m = CANDS[cid]; kind = m["kind"]
        if kind == "item":
            j_ev, y_ev = ev_by_kind.get("item", ([], []))
            if j_ev:
                al, k = B.infer_user(self.KM.Efull[j_ev], self.KM.bfull[j_ev], np.array(y_ev, float), self.KM.d)
            else:
                al, k = 0.0, np.zeros(self.KM.d)
            return float(B.sigmoid(self.KM.bfull[m["key"]] + al + self.KM.Efull[m["key"]] @ k))
        mf = self.mf.get(kind)
        if mf is None or m["key"] not in mf["erow"]:
            return 0.5
        keys, ys = ev_by_kind.get(kind, ([], []))
        rows = [mf["erow"][kk] for kk in keys if kk in mf["erow"]]
        yv = [yy for kk, yy in zip(keys, ys) if kk in mf["erow"]]
        if rows:
            al, k = B.infer_user(mf["E"][rows], mf["b"][rows], np.array(yv, float), B.D_MF)
        else:
            al, k = 0.0, np.zeros(B.D_MF)
        r = mf["erow"][m["key"]]
        return float(B.sigmoid(mf["b"][r] + al + mf["E"][r] @ k))


# =========================================================== the router (finding 3)
def taste_align(z, emb):
    nz = np.linalg.norm(z)
    if nz <= 0:
        return 0.5
    cs = float(z @ emb / (nz * np.linalg.norm(emb) + 1e-9))
    return 0.5 + 0.5 * max(min(cs, 1.0), -1.0)


def router_plan(env, CANDS, rec, V_pop, belief, s_anchor=None, trace=None):
    """Fold answers -> taste z_t; select argmax p_ans(q|events) * taste_align(z_t,q) * max(V_pop,0).
    s_anchor != None => tie-by-construction: follow s_anchor for the first WARMUP turns (E2), then
    descend. trace (list) collects per-turn picks for the mechanism readout."""
    FR, model = env["FR"], env["model"]
    allc = [m["cid"] for m in CANDS]
    z_cold = np.zeros(FR.W.shape[1])
    used = set(); plan = []; ev = []
    ev_by_kind = {"concept": ([], []), "attr": ([], []), "item": ([], [])}

    def record_event(cid, answered):
        m = CANDS[cid]; kind = m["kind"]
        if kind in ev_by_kind:
            ev_by_kind[kind][0].append(m["key"]); ev_by_kind[kind][1].append(1 if answered else 0)

    for t in range(TMAX):
        z = V2.fold_np_v2(FR, model, [tok_of(CANDS, rec, c) for c in ev],
                          [rec["nat_arr"][c] for c in ev if rec["nat_arr"][c] is not None]) if ev else z_cold
        if s_anchor is not None and t < WARMUP and t < len(s_anchor):
            bc = s_anchor[t]
            if bc in used:                              # already taken -> fall through to scoring
                bc = None
        else:
            bc = None
        if bc is None:
            best = -1e18
            for c in allc:
                if c in used:
                    continue
                m = CANDS[c]
                pa = belief.p_ans(CANDS, c, ev_by_kind)
                ta = taste_align(z, m["emb"])
                v = max(V_pop[c], 0.0) + 1e-4
                score = pa * ta * v
                if score > best:
                    best, bc = score, c
        if bc is None:
            break
        used.add(bc); m = CANDS[bc]
        answered = bool(rec["ans_arr"][bc])
        plan.append(bc if answered else None)
        if answered:
            ev.append(bc)
        record_event(bc, answered)
        if trace is not None:
            trace.append(dict(t=t, pick=f"{m['kind']}:{m['key']}", habitat=bool(m["habitat"]),
                              answered=answered, klev=int(rec["klev_arr"][bc]),
                              taste_align=round(taste_align(z, m["emb"]), 3),
                              p_ans=round(belief.p_ans(CANDS, bc, ev_by_kind), 3),
                              z_norm=round(float(np.linalg.norm(z)), 3)))
    return plan


# =========================================================== main
PREREG = (
    "## Pre-registered reads (printed BEFORE results)\n\n"
    "**PRIMARY (chosen plainly): endpoint NDCG@50 at T=24.** Rationale (audit Finding 5): anytime@10\n"
    "saturates by turn 4-6 and cannot pay back exploration; endpoint gives an adaptive descent room to\n"
    "register, and @50 reaches the niche/tail depth where per-user structure lives. All other cells\n"
    "(anytime@10, endpoint@10/@50 at T=12/24) are reported too, no cherry-pick.\n\n"
    "For each router, pooled (router - split-fair static) over the 4 split estimates' concatenated\n"
    "per-user deltas (seeds {0,1} x eval-half {A,B}), paired per-user bootstrap:\n"
    "- **(i) CI excludes 0 POSITIVE** = the first fair adaptivity win in a pool that CONTAINS the fuel.\n"
    "- **(ii) CI includes 0** = honest tie -- reported ONLY WITH its MDE (a tie below the MDE is a\n"
    "  non-answer, not evidence of no-effect).\n"
    "- **(iii) CI excludes 0 NEGATIVE** = the router loses even in the habitat pool (reported plainly).\n\n"
    "MECHANISM CHECK (pre-registered): per-arm mean per-user REFUSAL RATE in the habitat -- the router\n"
    "should show LOWER refusals than the static (it routes around no-clue questions). Plus the\n"
    "taste-descent trace for 3 example users (which habitat questions got picked after which answers).\n\n"
    "Anchoring (audit Finding 6): primary comparator = s-mixed-new (split-fair, habitat pool). Also\n"
    "reported vs s-item-new, and vs the OLD top-coverage-pool static (reference row).\n\n")


def _pool_report(CANDS, users, tag):
    """Answer-rate distribution + per-user in-pool refusal rate."""
    by_ch = collections.defaultdict(list)
    hab = collections.Counter()
    for m in CANDS:
        by_ch[m["kind"]].append(m["rate"])
        if m["habitat"]:
            hab[m["kind"]] += 1
    dist = {}
    for ch, rates in by_ch.items():
        r = np.array(rates)
        dist[ch] = dict(n=len(r), n_habitat=int(hab[ch]), rate_mean=float(r.mean()),
                        rate_min=float(r.min()), rate_p25=float(np.percentile(r, 25)),
                        rate_median=float(np.median(r)), rate_max=float(r.max()),
                        frac_in_band=float(np.mean((r >= HAB_LO) & (r <= HAB_HI))))
    # per-user in-pool refusal rate = fraction of pool candidates the user CANNOT answer
    ref = []
    nc = len(CANDS)
    for rec in users:
        ref.append(float(1.0 - rec["ans_arr"].sum() / nc))
    ref = np.array(ref)
    refusal = dict(mean=float(ref.mean()), p50=float(np.median(ref)), p90=float(np.percentile(ref, 90)),
                   max=float(ref.max()))
    print(f"  [pool {tag}] {nc} cands; per-channel answer-rate + habitat count:", flush=True)
    for ch in ("concept", "item", "attr"):
        if ch in dist:
            d = dist[ch]
            print(f"     {ch:8s} n={d['n']:3d} hab={d['n_habitat']:3d} rate[min/p25/med/max]="
                  f"{d['rate_min']:.2f}/{d['rate_p25']:.2f}/{d['rate_median']:.2f}/{d['rate_max']:.2f} "
                  f"in-band={d['frac_in_band']:.2f}", flush=True)
    print(f"     per-user in-pool REFUSAL rate: mean={refusal['mean']:.3f} p90={refusal['p90']:.3f} "
          f"max={refusal['max']:.3f}", flush=True)
    return dict(channels=dist, per_user_refusal=refusal, n_cands=nc)


def run():
    t0 = time.time()
    print(f"==== HABITAT EXPERIMENT [{ARENA.upper()} arena, answered iff k>={KTHR}] "
          f"(DIRECTIONAL 173/300) ====", flush=True)
    D, split, grid, users, memb, battery = A.load_env()
    FR = L.Frozen(D)
    model = V2.FoldV2(); blob = torch.load(CKPT, map_location="cpu")
    model.load_state_dict(blob["model"]); model.eval()
    env = dict(FR=FR, model=model, D=D)
    KM = SC_C.KmapItem(D)

    md(f"# HABITAT EXPERIMENT [{ARENA.upper()} ARENA] -- fuel-in-pool, fuel-trained router, "
       f"both metrics (v2 fold)\n\n", mode="w")
    if ARENA == "strict":
        md("**STRICT ARENA (author addition):** a question is ANSWERED iff knowledge == know_well "
           "(k>=2). A rough_idea response is a HEDGE: it consumes the turn, folds NO value token, and "
           "feeds y=0 to the online knowledge belief (the hedge still updates the knowledge map). "
           "Pool habitat band, value model, beliefs, and statics all use k>=2 answerability -- the "
           "answer rates below ARE the know_well rates, refusal-rich and user-specific everywhere.\n\n")
    md("> **DIRECTIONAL ONLY -- 173/300 users, answerer-v1 working grid NOT frozen; v2 fold "
       "`.cache/i25_fold_v2_best.pt`. Re-run on the frozen 300-user grid before any citation.**\n\n"
       f"Date 2026-07-09. Script `scripts/habitat_experiment.py`. NO LLM API calls; local compute. "
       f"{len(users)} users; fold val {blob['state']['best_val']:.4f}. Paired per-user bootstrap "
       f"BOOT={BOOT}. Executes the audit's single corrective experiment (experiments/FRESH_AUDIT.md), "
       f"removing masks 1 (pool), 2 (fuel-free training), 3 (no taste-policy) at once.\n\n")

    # ---- E7 arm-symmetry table (printed before results) ----
    sym = [
        ("arena answerability", f"{ARENA}: answered iff k>={KTHR}" +
         ("; rough_idea = hedge (turn burned, no token, y=0 to belief)" if KTHR == 2 else ""),
         "identical semantics every arm"),
        ("candidate pool", "habitat-stratified (20 top-cov + 40 rate-0.2..0.7), population-selected", "SAME pool for every arm"),
        ("static construction", "greedy on CONSTRUCTION half only (endpoint@50)", "router beliefs also construction half only"),
        ("value model V(q)", "mean single-answer lift on CONSTRUCTION half (REAL grid)", "NOT population sims (Finding 2 fix)"),
        ("knowledge belief", "LOUO logistic-MF on CONSTRUCTION half (concept/attr) + population kmap (item)", "eval user's own cells never train it (E5)"),
        ("online evidence", "eval user's own answered/refused events only", "same for router; static is non-adaptive"),
        ("selection value peek", "router uses V_pop (population), NEVER the eval user's arena answer value", "Finding 9 fixed"),
        ("eval set", "eval half only, every user every arm (E1 survivorship)", "refusal = no-op turn, cold fallback"),
        ("metrics", "anytime@10 + endpoint@10/@50 at T=12/24; PRIMARY endpoint@50 T=24", "same ruler every arm"),
    ]
    md("## E7 arm-symmetry table (what each arm may see / learn -- listed before results)\n\n"
       "| dimension | specification | symmetry note |\n|---|---|---|\n")
    for a, b, c in sym:
        md(f"| {a} | {b} | {c} |\n")
    md("\n")
    print("  E7 arm-symmetry table written.", flush=True)
    md(PREREG); print("\n" + PREREG, flush=True)

    # ---- build pools + report distributions ----
    print("==== POOL CONSTRUCTION ====", flush=True)
    CANDS, _ = build_pool(D, FR, users, memb, battery, habitat=True)
    new_report = _pool_report(CANDS, users, "NEW habitat")
    # old pool (reference) -- separate CANDS/arrays; keep users' arrays for NEW, so use a shallow copy set
    old_users = [dict(rec) for rec in users]
    CANDS_OLD, _ = build_pool(D, FR, old_users, memb, battery, habitat=False)
    old_report = _pool_report(CANDS_OLD, old_users, "OLD top-cov")

    md("## Pool answer-rate distribution + per-user in-pool refusal rate\n\n"
       "| pool | channel | n | habitat | rate min | p25 | median | max | frac in 0.2-0.7 band |\n"
       "|---|---|--:|--:|--:|--:|--:|--:|--:|\n")
    for tag, rep in (("NEW habitat", new_report), ("OLD top-cov", old_report)):
        for ch in ("concept", "item", "attr"):
            if ch in rep["channels"]:
                d = rep["channels"][ch]
                md(f"| {tag} | {ch} | {d['n']} | {d['n_habitat']} | {d['rate_min']:.2f} | "
                   f"{d['rate_p25']:.2f} | {d['rate_median']:.2f} | {d['rate_max']:.2f} | "
                   f"{d['frac_in_band']:.2f} |\n")
    md("\n**Per-user in-pool REFUSAL rate (fraction of pool the user cannot answer):**\n\n"
       "| pool | mean | median | p90 | max |\n|---|--:|--:|--:|--:|\n")
    for tag, rep in (("NEW habitat", new_report), ("OLD top-cov", old_report)):
        r = rep["per_user_refusal"]
        md(f"| {tag} | {r['mean']:.3f} | {r['p50']:.3f} | {r['p90']:.3f} | {r['max']:.3f} |\n")
    md("\nThe OLD pool's per-user refusal is ~0 (audit Finding 1: answerability is the constant 1, "
       "nothing to route on). The NEW pool restores real refusal variance -- the fuel is now in the "
       "action space.\n\n")

    item_pool = [m["cid"] for m in CANDS if m["kind"] == "item"]
    mixed_pool = [m["cid"] for m in CANDS]
    old_mixed = [m["cid"] for m in CANDS_OLD]

    # ---- 4 split estimates ----
    n = len(users)
    ARMS = ["s-item-new", "s-mixed-new", "s-mixed-old", "r-hab-anchored", "r-hab-free"]
    # pooled per-user vectors: pooled[arm][(K,T)] endpoints + ("any10",T) anytime@10; refusal per arm
    METRIC_KEYS = [(kk, T) for kk in KS for T in BUDGETS] + [("any10", T) for T in BUDGETS]
    pooled = {arm: {mk: [] for mk in METRIC_KEYS} for arm in ARMS}
    pooled_ref = {arm: [] for arm in ARMS}
    est_summ = {}
    traces = None

    for seed in (0, 1):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n); half = n // 2
        halves = {"A": sorted(perm[:half].tolist()), "B": sorted(perm[half:].tolist())}
        for evalhalf in ("B", "A"):
            constr = "A" if evalhalf == "B" else "B"
            ev_users = [users[i] for i in halves[evalhalf]]
            cn_users = [users[i] for i in halves[constr]]
            ev_old = [old_users[i] for i in halves[evalhalf]]
            cn_old = [old_users[i] for i in halves[constr]]
            cn_cold50 = cold_k(FR, cn_users, PRIMARY_K)
            cn_cold50_old = cold_k(FR, cn_old, PRIMARY_K)
            tag = f"seed{seed}_eval{evalhalf}"
            print(f"\n---- estimate {tag}: construct on {constr} (n={len(cn_users)}), eval on "
                  f"{evalhalf} (n={len(ev_users)}) ----", flush=True)

            # split-fair statics on construction half (greedy endpoint@50)
            print("  building split-fair statics (greedy endpoint@50, no cov cap) ...", flush=True)
            s_item = build_greedy(env, CANDS, item_pool, cn_users, cn_cold50, TMAX, PRIMARY_K)
            s_mixed = build_greedy(env, CANDS, mixed_pool, cn_users, cn_cold50, TMAX, PRIMARY_K)
            s_old = build_greedy(env, CANDS_OLD, old_mixed, cn_old, cn_cold50_old, TMAX, PRIMARY_K)

            # fuel-trained value + belief on construction half (Findings 2)
            print("  fitting value V_pop (real grid, construction half) + LOUO-MF beliefs ...", flush=True)
            V_pop = fit_value_pop(env, CANDS, cn_users, PRIMARY_K)
            belief = Belief(cn_users, CANDS, KM)

            # routers on eval users
            print("  running routers (anchored + free) ...", flush=True)
            r_anch = [router_plan(env, CANDS, rec, V_pop, belief, s_anchor=s_mixed) for rec in ev_users]
            r_free = [router_plan(env, CANDS, rec, V_pop, belief, s_anchor=None) for rec in ev_users]
            tie_ok = all(r_anch[i][0] == (s_mixed[0] if ev_users[i]["ans_arr"][s_mixed[0]] else None)
                         or r_anch[i][0] == s_mixed[0] for i in range(0, len(ev_users), 20))

            # collect traces (first estimate only, 3 users)
            if traces is None:
                traces = {}
                for label, anchor in (("r-hab-free", None), ("r-hab-anchored", s_mixed)):
                    ex = []
                    for i in range(min(3, len(ev_users))):
                        tr = []
                        router_plan(env, CANDS, ev_users[i], V_pop, belief, s_anchor=anchor, trace=tr)
                        ex.append(dict(user=int(ev_users[i]["u"]), trace=tr[:TMAX]))
                    traces[label] = ex

            # evaluate every arm on eval users
            plan_map = {
                "s-item-new": static_plans(CANDS, s_item, ev_users, TMAX),
                "s-mixed-new": static_plans(CANDS, s_mixed, ev_users, TMAX),
                "r-hab-anchored": r_anch,
                "r-hab-free": r_free,
            }
            e = {"n_eval": len(ev_users), "tie_ok": bool(tie_ok)}
            arm_pt = {}
            for arm, plans in plan_map.items():
                pt, ref = eval_plan_metrics(env, CANDS, plans, ev_users, TMAX)
                arm_pt[arm] = pt
                pooled_ref[arm].extend(list(ref)); e.setdefault("refusal", {})[arm] = float(np.mean(ref))
                for kk in KS:
                    for T in BUDGETS:
                        pooled[arm][(kk, T)].extend(list(pt[kk][:, T - 1]))
                for T in BUDGETS:
                    pooled[arm][("any10", T)].extend(list(pt[10][:, :T].mean(axis=1)))
            # old-pool reference static (its own CANDS/arrays)
            pt_old, ref_old = eval_plan_metrics(env, CANDS_OLD,
                                                static_plans(CANDS_OLD, s_old, ev_old, TMAX), ev_old, TMAX)
            pooled_ref["s-mixed-old"].extend(list(ref_old)); e.setdefault("refusal", {})["s-mixed-old"] = float(np.mean(ref_old))
            for kk in KS:
                for T in BUDGETS:
                    pooled["s-mixed-old"][(kk, T)].extend(list(pt_old[kk][:, T - 1]))
            for T in BUDGETS:
                pooled["s-mixed-old"][("any10", T)].extend(list(pt_old[10][:, :T].mean(axis=1)))

            # per-estimate primary deltas
            for arm in ("r-hab-anchored", "r-hab-free"):
                cb = paired(list(arm_pt[arm][PRIMARY_K][:, PRIMARY_T - 1]),
                            list(arm_pt["s-mixed-new"][PRIMARY_K][:, PRIMARY_T - 1]))
                e.setdefault("primary_vs_smixed", {})[arm] = cb
                print(f"    {arm:16s} endpoint@50 T24 vs s-mixed-new {cb['delta']:+.4f}"
                      f"[{cb['ci'][0]:+.4f},{cb['ci'][1]:+.4f}] (MDE {cb['mde']:.4f})", flush=True)
            print(f"    refusal: " + " ".join(f"{a}={e['refusal'][a]:.3f}" for a in
                  ("s-mixed-new", "r-hab-anchored", "r-hab-free")) + f" | tie@t0={tie_ok}", flush=True)
            est_summ[tag] = e
            _checkpoint(est_summ)

    # ---- pooled results ----
    def endp_pool(arm, kk, T):
        return pooled[arm][(kk, T)]

    results = dict(banner=f"DIRECTIONAL 173/300, grid unfrozen (v2 fold); ARENA={ARENA} (k>={KTHR})",
                   arena=ARENA, kthr=KTHR,
                   n_users=n, pool_new=new_report, pool_old=old_report,
                   primary=dict(K=PRIMARY_K, T=PRIMARY_T), estimates=est_summ, traces=traces)

    # arm means (all metrics, pooled)
    arm_means = {}
    for arm in ARMS:
        arm_means[arm] = {f"K{kk}_T{T}": float(np.mean(pooled[arm][(kk, T)])) for kk in KS for T in BUDGETS}
        for T in BUDGETS:
            arm_means[arm][f"any10_T{T}"] = float(np.mean(pooled[arm][("any10", T)]))
        arm_means[arm]["refusal"] = float(np.mean(pooled_ref[arm]))
    results["arm_means"] = arm_means

    # pooled contrasts vs s-mixed-new and vs s-item-new, every metric
    contrasts = {}
    for arm in ("r-hab-anchored", "r-hab-free"):
        for base in ("s-mixed-new", "s-item-new"):
            for kk in KS:
                for T in BUDGETS:
                    cb = paired(endp_pool(arm, kk, T), endp_pool(base, kk, T))
                    contrasts[f"{arm}__vs__{base}__K{kk}_T{T}"] = cb
            for T in BUDGETS:
                cb = paired(pooled[arm][("any10", T)], pooled[base][("any10", T)])
                contrasts[f"{arm}__vs__{base}__any10_T{T}"] = cb
    # anytime@10 pooled (needs full curves; approximate with endpoint sequence unavailable -> recompute
    # from per-estimate? we stored only endpoints. anytime is reported per-estimate mean instead.)
    results["contrasts"] = contrasts

    # refusal mechanism contrast (router vs static)
    ref_contrast = {}
    for arm in ("r-hab-anchored", "r-hab-free"):
        ref_contrast[arm] = paired(pooled_ref[arm], pooled_ref["s-mixed-new"])
    results["refusal_mechanism"] = ref_contrast

    prim = contrasts[f"r-hab-anchored__vs__s-mixed-new__K{PRIMARY_K}_T{PRIMARY_T}"]
    prim_free = contrasts[f"r-hab-free__vs__s-mixed-new__K{PRIMARY_K}_T{PRIMARY_T}"]

    def verdict(cb):
        lo, hi = cb["ci"]
        if lo > 0:
            return "FAIR-WIN (CI excl 0, +): first fair adaptivity win in a fuel-containing pool"
        if hi < 0:
            return "FAIR-LOSS (CI excl 0, -): router loses even in the habitat pool"
        tievn = "below MDE (non-answer)" if abs(cb["delta"]) < cb["mde"] else "above MDE"
        return f"TIE (CI incl 0), {tievn}"

    results["verdicts"] = dict(primary_anchored=verdict(prim), primary_free=verdict(prim_free))
    results["wall_min"] = round((time.time() - t0) / 60, 2)

    _write_results_md(results, arm_means, contrasts, ref_contrast, traces, est_summ)
    json.dump(results, open(OUT_JSON, "w"), indent=1, default=str)

    print("\n==== POOLED PRIMARY VERDICT (endpoint@50, T=24) ====", flush=True)
    print(f"  r-hab-anchored - s-mixed-new = {prim['delta']:+.4f}[{prim['ci'][0]:+.4f},{prim['ci'][1]:+.4f}]"
          f"  MDE={prim['mde']:.4f} (n={prim['n']}) -> {results['verdicts']['primary_anchored']}", flush=True)
    print(f"  r-hab-free     - s-mixed-new = {prim_free['delta']:+.4f}[{prim_free['ci'][0]:+.4f},"
          f"{prim_free['ci'][1]:+.4f}]  MDE={prim_free['mde']:.4f} -> {results['verdicts']['primary_free']}",
          flush=True)
    for arm in ("r-hab-anchored", "r-hab-free"):
        rc = ref_contrast[arm]
        print(f"  refusal {arm} - s-mixed-new = {rc['delta']:+.4f}[{rc['ci'][0]:+.4f},{rc['ci'][1]:+.4f}] "
              f"(negative = router refuses LESS)", flush=True)
    print(f"\n[done] wrote {OUT_MD} + {OUT_JSON} (wall {results['wall_min']}m)", flush=True)
    return results


def _checkpoint(est_summ):
    json.dump(dict(banner="DIRECTIONAL 173/300 checkpoint", estimates=est_summ),
              open(CKPT_JSON, "w"), indent=1, default=str)


def _write_results_md(results, arm_means, contrasts, ref_contrast, traces, est_summ):
    md("## Results\n\n### Arm means (pooled over 4 estimates)\n\n"
       "| arm | any@10 T24 | end@10 T12 | end@10 T24 | end@50 T12 | **end@50 T24 (PRIMARY)** | refusal |\n"
       "|---|--:|--:|--:|--:|--:|--:|\n")
    for arm in ("s-item-new", "s-mixed-new", "s-mixed-old", "r-hab-anchored", "r-hab-free"):
        m = arm_means[arm]
        md(f"| {arm} | {m['any10_T24']:.4f} | {m['K10_T12']:.4f} | {m['K10_T24']:.4f} | {m['K50_T12']:.4f} | "
           f"**{m['K50_T24']:.4f}** | {m['refusal']:.3f} |\n")
    md("\n(anytime@10 saturates by t4-6 -- reported for completeness; endpoint@50 T24 is the pre-"
       "registered primary. All three metric families reported per audit Finding 5.)\n\n")

    md("### Pre-registered contrasts (pooled, paired bootstrap, with MDE)\n\n"
       "| router | vs | metric | delta [95% CI] | MDE | n | verdict |\n|---|---|---|---|--:|--:|---|\n")
    for arm in ("r-hab-anchored", "r-hab-free"):
        for base in ("s-mixed-new", "s-item-new"):
            for key, lbl in [("K50_T24", "end@50 T24"), ("K50_T12", "end@50 T12"),
                             ("K10_T24", "end@10 T24"), ("K10_T12", "end@10 T12"),
                             ("any10_T24", "any@10 T24"), ("any10_T12", "any@10 T12")]:
                cb = contrasts[f"{arm}__vs__{base}__{key}"]
                lo, hi = cb["ci"]
                v = "WIN" if lo > 0 else ("LOSS" if hi < 0 else ("tie<MDE" if abs(cb["delta"]) < cb["mde"] else "tie"))
                star = " **(PRIMARY)**" if (base == "s-mixed-new" and key == "K50_T24") else ""
                md(f"| {arm} | {base} | {lbl}{star} | {cb['delta']:+.4f}[{lo:+.4f},{hi:+.4f}] | "
                   f"{cb['mde']:.4f} | {cb['n']} | {v} |\n")
    md("\n")

    md("### Mechanism check -- per-user refusal rate (router should refuse LESS than the static)\n\n"
       "| arm | mean refusal rate | router - static [95% CI] |\n|---|--:|---|\n")
    md(f"| s-mixed-new (static) | {arm_means['s-mixed-new']['refusal']:.3f} | (baseline) |\n")
    for arm in ("r-hab-anchored", "r-hab-free"):
        rc = ref_contrast[arm]
        md(f"| {arm} | {arm_means[arm]['refusal']:.3f} | {rc['delta']:+.4f}[{rc['ci'][0]:+.4f},"
           f"{rc['ci'][1]:+.4f}] |\n")
    md("\nNegative router-minus-static = the router spends fewer turns on no-clue questions (the "
       "answerability-routing mechanism the habitat pool finally makes possible).\n\n")

    md("### Verdicts\n\n"
       f"- **PRIMARY (endpoint@50, T=24), r-hab-anchored vs s-mixed-new:** {results['verdicts']['primary_anchored']}.\n"
       f"- r-hab-free vs s-mixed-new (same metric): {results['verdicts']['primary_free']}.\n\n")

    md("### Taste-descent traces (3 example users; the interpretability readout)\n\n")
    for label in ("r-hab-free", "r-hab-anchored"):
        md(f"**{label}:**\n\n")
        for ex in traces.get(label, []):
            md(f"- user {ex['user']}: ")
            steps = []
            for s in ex["trace"][:12]:
                mark = "H" if s["habitat"] else "."
                a = "ans" if s["answered"] else "REF"
                steps.append(f"t{s['t']}[{mark}]{s['pick']}({a},k{s['klev']},ta={s['taste_align']},"
                             f"pa={s['p_ans']})")
            md("; ".join(steps) + "\n")
        md("\n")

    md("### Per-estimate detail (endpoint@50 T24 vs s-mixed-new)\n\n"
       "| estimate | n_eval | r-hab-anchored | r-hab-free | tie@t0 |\n|---|--:|---|---|---|\n")
    for tag, e in est_summ.items():
        pa = e.get("primary_vs_smixed", {})
        def fmt(a):
            cb = pa.get(a)
            return f"{cb['delta']:+.4f}[{cb['ci'][0]:+.4f},{cb['ci'][1]:+.4f}]" if cb else "-"
        md(f"| {tag} | {e['n_eval']} | {fmt('r-hab-anchored')} | {fmt('r-hab-free')} | {e['tie_ok']} |\n")
    md("\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--do", choices=["all"], default="all")
    ap.add_argument("--arena", choices=["lenient", "strict"], default="lenient")
    args = ap.parse_args()
    configure(args.arena)
    run()
