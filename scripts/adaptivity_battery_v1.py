"""adaptivity_battery_v1.py -- DIRECTIONAL ADAPTIVITY BATTERY on the PARTIAL (173-user) Answerer v1
environment. Author-directed 2026-07-08. NO LLM API calls; all local compute on cached grids.

*** DIRECTIONAL ONLY: 173/300 users (163 shards + 10 grid10); grid NOT frozen. Every output carries
    this flag. Do NOT cite as a final number. ***

Environment = the JUDGED answerer-v1 grid (channels concept/attribute/item; compact cells with
knowledge in {no_clue,rough_idea,know_well}, graded 4-level value, half-star rating, conf). Rated
known-half item cells are source="data" (real rating, know_well). Answerability = knowledge != no_clue
(k>=1); strong knowledge = know_well (k>=2). Fold weight: know_well=1.0, rough_idea=w_rough=0.5.

ERROR LEDGER honored (E1 survivorship: same full user set every arm/metric, refusal=no-op turn;
E2 tie-by-construction router; E3 privilege labelling; E4 fold canary gate; E5 circularity firewall
LEAVE-ONE-USER-OUT / population only, never the eval user's own cells for beliefs; E6 pre-registered
thresholds printed before results, paired per-user bootstrap CIs, deterministic seeds, ASCII).

Stages:  A fuel gates (heterogeneity ICC / validity gap / knowledge scarcity)   -- no fold
         B discovery feasibility (AUC-vs-t + split-half asymptote, 3 beliefs)   -- kmap + LOUO MF
         C prize decomposition (mini-arena, T=12, statics / u-table / r-blind)  -- I2.5 fold

Run:  python scripts/adaptivity_battery_v1.py --stage a|b|c   (a writes header first)
"""
import os, sys, json, time, argparse, collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings("ignore")
import llm_answerability_gate as G

CACHE = ".cache/instrument2"
SHARDS = [f"{CACHE}/answerer_v1_shard{k}.json" for k in range(4)]
GRID10 = f"{CACHE}/answerer_v1_grid10.json"
WORKING = f"{CACHE}/answerer_v1_grid173_WORKING.json"      # NOT-FROZEN merged copy
TAG_MEMB = f"{CACHE}/tag_membership.json"
BATTERY = f"{CACHE}/attr_battery_500.json"
OUT_MD = "experiments/ADAPTIVITY_BATTERY_V1.md"
OUT_JSON = "experiments/adaptivity_battery_v1.json"

KIDX = {"no_clue": 0, "rough_idea": 1, "know_well": 2}
CENTERED_FOLD = {"hated": -1.0, "meh": -1.0 / 3.0, "liked": 1.0 / 3.0, "loved": 1.0}
W_ROUGH = 0.5
FOLD_WEIGHT = {"know_well": 1.0, "rough_idea": W_ROUGH, "no_clue": 0.0}
BOOT = 5000
SEED = 0
RATEDNESS_CEILING = 0.725      # the OLD arena rated-ness AUC ceiling (STATE_2026-07-08 Q&A #1)

BANNER = ("> **DIRECTIONAL ONLY** -- 173/300 users (163 shards + 10 grid10 merged in-memory); the "
          "answerer-v1 grid is NOT frozen. Every number below is provisional and must be re-run on the "
          "frozen 300-user grid before any citation.\n\n")


# ============================================================ merge (E5: environment = judged grid only)
def merge_grid():
    users = {}
    origin = {}
    g10 = json.load(open(GRID10))["users"]
    for u, rec in g10.items():
        users[u] = rec; origin[u] = "grid10"
    for sp in SHARDS:
        b = json.load(open(sp))
        for u, rec in b.get("users", {}).items():
            if u in users:
                continue                                     # keep first (grid10 wins)
            users[u] = rec; origin[u] = os.path.basename(sp)
    json.dump({"NOT_FROZEN": "DIRECTIONAL working merge of 163 shard users + 10 grid10; 173/300; "
               "grid unfrozen -- do not cite", "split_seed": 123, "n_users": len(users), "users": users},
              open(WORKING, "w"))
    print(f"[merge] {len(users)} users -> {WORKING} (NOT FROZEN)", flush=True)
    return users


# ============================================================ common env loader
def cell_know(ans):
    """knowledge index 0/1/2 or None (unparsed)."""
    if not ans or not isinstance(ans, dict):
        return None
    return KIDX.get(ans.get("knowledge"))


def cell_foldval(ans):
    """4-level centered fold value * fold weight (concepts/attrs)."""
    k = ans.get("knowledge")
    if k == "no_clue" or "value" not in ans:
        return 0.0
    return CENTERED_FOLD.get(ans.get("value"), 0.0) * FOLD_WEIGHT.get(k, 0.0)


def load_env():
    """Returns D, split, merged grid, and per-user recs (known/like/tlike/prof/cmean + channel cells).
    User filter mirrors i25_phase4.assemble (like, tlike, known>=4) so fold recs are comparable."""
    D = G.load_data()
    split = G.build_split(D)
    grid = merge_grid()
    memb = json.load(open(TAG_MEMB))["membership"]           # tagId(str) -> [dense ids]
    battery = {e["entity_id"]: e for e in json.load(open(BATTERY))["entities"]}

    users = []
    for us, rec in grid.items():
        u = int(us)
        if u not in split:
            continue
        kn, ho = split[u]
        rat = dict(D["rat_by_u"][u])
        known = {j: rat[j] for j in kn if j in rat}
        like = [j for j in kn if rat.get(j, 0) >= 4]
        tlike = set(j for j in ho if rat.get(j, 0) >= 4)
        if not like or not tlike or len(known) < 4:
            continue
        cm = float(np.mean(list(known.values())))
        concept, attribute, item_llm, item_data = [], [], [], []
        for i, c in rec["Q"].items():
            ans = c.get("ans")
            k = cell_know(ans)
            ch = c["channel"]
            if ch == "concept":
                tid = int(c["tagId"])
                concept.append(dict(tagId=tid, k=k, ans=ans,
                                    pop=len(memb.get(str(tid), []))))
            elif ch == "attribute":
                eid = c["entity_id"]
                attribute.append(dict(eid=eid, atype=c.get("atype"), k=k, ans=ans,
                                      nmov=(battery.get(eid, {}) or {}).get("n_movies", 0)))
            elif ch == "item":
                j = int(c["j"])
                item_llm.append(dict(j=j, k=k, ans=ans, cnt=float(D["cnt"][j]),
                                     pr=float(D["pr"][j]), tier=str(D["tier"][j])))
        for jstr, c in rec.get("data", {}).items():
            j = int(c["j"])
            item_data.append(dict(j=j, stars=float(c["stars"]), cnt=float(D["cnt"][j]),
                                  pr=float(D["pr"][j]), tier=str(D["tier"][j])))
        users.append(dict(u=u, known=known, like=like, tlike=tlike, prof=set(kn), cmean=cm,
                          concept=concept, attribute=attribute, item_llm=item_llm, item_data=item_data,
                          origin_dom=rec.get("dom")))
    print(f"[env] {len(users)} of {len(grid)} merged users pass the fold filter (like/tlike/known>=4)",
          flush=True)
    return D, split, grid, users, memb, battery


# ============================================================ stats helpers
def paired_boot(d, seed=SEED):
    d = np.asarray([x for x in d if x is not None and not (isinstance(x, float) and np.isnan(x))], float)
    if len(d) == 0:
        return dict(delta=float("nan"), ci=[float("nan"), float("nan")], n=0)
    rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(BOOT)])
    return dict(delta=float(d.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                n=int(len(d)))


def icc_oneway(y, groups, cov=None, seed=SEED):
    """One-way random-effects ICC of y across groups (users), AFTER partialling out a fixed covariate
    cov (e.g. log-popularity) by OLS. Returns ICC point + user-bootstrap CI + base rate."""
    y = np.asarray(y, float)
    if cov is not None:
        X = np.column_stack([np.ones(len(y)), np.asarray(cov, float)])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ beta
    else:
        r = y - y.mean()
    gids = np.asarray(groups)

    def _icc(mask):
        rr = r[mask]; gg = gids[mask]
        uq = np.unique(gg)
        k = len(uq)
        N = len(rr)
        if k < 2 or N <= k:
            return float("nan")
        grand = rr.mean()
        ssb = ssw = 0.0
        sizes = []
        for g_ in uq:
            v = rr[gg == g_]; sizes.append(len(v))
            ssb += len(v) * (v.mean() - grand) ** 2
            ssw += ((v - v.mean()) ** 2).sum()
        sizes = np.array(sizes)
        msb = ssb / (k - 1); msw = ssw / (N - k)
        n0 = (N - (sizes ** 2).sum() / N) / (k - 1)
        icc = (msb - msw) / (msb + (n0 - 1) * msw) if (msb + (n0 - 1) * msw) > 0 else 0.0
        return max(icc, 0.0)

    point = _icc(np.ones(len(r), bool))
    # user-cluster bootstrap
    uq = np.unique(gids)
    rng = np.random.default_rng(seed)
    idx_by_g = {g_: np.where(gids == g_)[0] for g_ in uq}
    bs = []
    for _ in range(400):
        samp = rng.choice(uq, len(uq), replace=True)
        mask_idx = np.concatenate([idx_by_g[g_] for g_ in samp])
        # rebuild r/gids for the resample with relabelled groups (avoid collisions)
        rr = r[mask_idx]
        gg = np.concatenate([np.full(len(idx_by_g[g_]), n) for n, g_ in enumerate(samp)])
        uq2 = np.unique(gg); k = len(uq2); N = len(rr)
        if k < 2 or N <= k:
            continue
        grand = rr.mean(); ssb = ssw = 0.0; sizes = []
        for g_ in uq2:
            v = rr[gg == g_]; sizes.append(len(v))
            ssb += len(v) * (v.mean() - grand) ** 2; ssw += ((v - v.mean()) ** 2).sum()
        sizes = np.array(sizes); msb = ssb / (k - 1); msw = ssw / (N - k)
        n0 = (N - (sizes ** 2).sum() / N) / (k - 1)
        den = msb + (n0 - 1) * msw
        bs.append(max((msb - msw) / den, 0.0) if den > 0 else 0.0)
    ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if bs else [float("nan")] * 2
    return dict(icc=float(point), ci=ci, n_obs=int(len(r)), n_users=int(len(uq)))


_CGVEC = {}
def concept_gvec(D, memb, tagId):
    """Genre vector of a genome tag = mean Gmat over its member items (via tag_membership)."""
    if tagId in _CGVEC:
        return _CGVEC[tagId]
    mems = [j for j in memb.get(str(tagId), []) if 0 <= j < int(D["ni"])]
    v = D["Gmat"][mems].astype(np.float64).mean(0) if mems else np.zeros(len(G.GENRES))
    _CGVEC[tagId] = v
    return v


def md_write(txt, mode="a"):
    os.makedirs("experiments", exist_ok=True)
    open(OUT_MD, mode, encoding="utf-8").write(txt)


# ============================================================ STAGE A -- fuel gates
def stage_a():
    D, split, grid, users, memb, battery = load_env()
    print("\n==== STAGE A -- FUEL GATES (DIRECTIONAL 173/300) ====", flush=True)
    print("PRE-REGISTERED THRESHOLDS (printed before results):", flush=True)
    print("  A1 heterogeneity: ICC >= 0.05 (beyond log-popularity), per channel, k>=1 and k>=2.", flush=True)
    print("  A2 validity gap : user-specific-signal gap > 0 with CI excl 0 in the MODERATE band.", flush=True)
    print("  A3 scarcity     : locate strata with answer-rate in the 0.2-0.7 adaptivity habitat.", flush=True)

    out = {"banner": "DIRECTIONAL 173/300, grid unfrozen"}

    # ---------- A1 heterogeneity ICC ----------
    chan_rows = {}
    for ch, cells_key, pop_fn in (
            ("concept", "concept", lambda c: np.log(c["pop"] + 1.0)),
            ("attribute", "attribute", lambda c: np.log(c["nmov"] + 1.0)),
            ("item", "item_llm", lambda c: np.log(c["cnt"] + 1.0))):
        y1, y2, cov, grp = [], [], [], []
        for rec in users:
            for c in rec[cells_key]:
                if c["k"] is None:
                    continue
                y1.append(1.0 if c["k"] >= 1 else 0.0)
                y2.append(1.0 if c["k"] >= 2 else 0.0)
                cov.append(pop_fn(c)); grp.append(rec["u"])
        r1 = icc_oneway(y1, grp, cov); r2 = icc_oneway(y2, grp, cov)
        r1raw = icc_oneway(y1, grp, None); r2raw = icc_oneway(y2, grp, None)
        chan_rows[ch] = dict(k1=r1, k2=r2, k1_raw=r1raw, k2_raw=r2raw,
                             base_k1=float(np.mean(y1)), base_k2=float(np.mean(y2)), n=len(y1))
        print(f"  [A1 {ch:9s}] base k>=1 {np.mean(y1):.3f} k>=2 {np.mean(y2):.3f} | "
              f"ICC(k>=1|logpop)={r1['icc']:.3f}[{r1['ci'][0]:.3f},{r1['ci'][1]:.3f}] "
              f"ICC(k>=2|logpop)={r2['icc']:.3f}[{r2['ci'][0]:.3f},{r2['ci'][1]:.3f}] (n={len(y1)})",
              flush=True)
    out["A1"] = chan_rows

    # ---------- A2 validity gap (user-specific taste-proximity signal, per popularity band) ----------
    # The top-800 item bank is entirely in the GLOBAL 'famous' tier (pr>=0.95), so the design's
    # "moderate band" is realized as WITHIN-BANK popularity terciles (low/mid/high cnt); mid = the
    # moderate band. Non-circular user-specific-signal test: within a band, answerability(k>=1) of
    # taste-NEAR items (genre matches the user's known-liked taste, above per-user median) minus
    # taste-FAR items. gap = rate(near) - rate(far). Concept channel gets the same test (spans the
    # answerability range across all 1128 tags). k>=2 variant reported too (item k>=1 is near-ceiling).
    all_cnt = np.array([c["cnt"] for rec in users for c in rec["item_llm"]])
    iq = np.quantile(all_cnt, [1 / 3, 2 / 3]) if len(all_cnt) else [0, 0]

    def band_of(cnt):
        return "low" if cnt <= iq[0] else ("mid" if cnt <= iq[1] else "high")

    def taste_vec(rec):
        g = np.zeros(len(G.GENRES))
        for j in rec["like"]:
            g = g + D["Gmat"][j]
        n = np.linalg.norm(g)
        return (g / n) if n > 0 else None

    a2 = {"item_bands": {}, "concept": {}}
    for band in ("low", "mid", "high"):
        nf1, nf2, rated_gap = [], [], []
        for rec in users:
            gu = taste_vec(rec)
            if gu is None:
                continue
            cells = [c for c in rec["item_llm"] if band_of(c["cnt"]) == band and c["k"] is not None]
            if len(cells) < 6:
                continue
            gm = np.array([float(gu @ D["Gmat"][c["j"]] /
                                 (np.linalg.norm(D["Gmat"][c["j"]]) + 1e-9)) for c in cells])
            y1 = np.array([1.0 if c["k"] >= 1 else 0.0 for c in cells])
            y2 = np.array([1.0 if c["k"] >= 2 else 0.0 for c in cells])
            med = np.median(gm)
            n_m, f_m = gm >= med, gm < med
            if n_m.any() and f_m.any():
                nf1.append(float(y1[n_m].mean() - y1[f_m].mean()))
                nf2.append(float(y2[n_m].mean() - y2[f_m].mean()))
                rated_gap.append(float(1.0 - y1.mean()))
        a2["item_bands"][band] = dict(taste_near_far_k1=paired_boot(nf1),
                                      taste_near_far_k2=paired_boot(nf2),
                                      rated_vs_llm_ref=paired_boot(rated_gap))
    # concept taste-proximity gap (all tags)
    cnf1, cnf2 = [], []
    for rec in users:
        gu = taste_vec(rec)
        if gu is None:
            continue
        cells = [c for c in rec["concept"] if c["k"] is not None]
        if len(cells) < 20:
            continue
        cg = np.array([concept_gvec(D, memb, c["tagId"]) for c in cells])
        gm = np.array([float(gu @ v / (np.linalg.norm(v) + 1e-9)) if np.linalg.norm(v) > 0 else 0.0
                       for v in cg])
        y1 = np.array([1.0 if c["k"] >= 1 else 0.0 for c in cells])
        y2 = np.array([1.0 if c["k"] >= 2 else 0.0 for c in cells])
        med = np.median(gm)
        n_m, f_m = gm >= med, gm < med
        if n_m.any() and f_m.any():
            cnf1.append(float(y1[n_m].mean() - y1[f_m].mean()))
            cnf2.append(float(y2[n_m].mean() - y2[f_m].mean()))
    a2["concept"] = dict(taste_near_far_k1=paired_boot(cnf1), taste_near_far_k2=paired_boot(cnf2))
    for band in ("low", "mid", "high"):
        b1 = a2["item_bands"][band]["taste_near_far_k1"]; b2 = a2["item_bands"][band]["taste_near_far_k2"]
        print(f"  [A2 item-{band:4s}] near-far k>=1 {b1['delta']:+.3f}[{b1['ci'][0]:+.3f},{b1['ci'][1]:+.3f}]"
              f" k>=2 {b2['delta']:+.3f}[{b2['ci'][0]:+.3f},{b2['ci'][1]:+.3f}] (n={b1['n']})", flush=True)
    cb1 = a2["concept"]["taste_near_far_k1"]; cb2 = a2["concept"]["taste_near_far_k2"]
    print(f"  [A2 concept ] near-far k>=1 {cb1['delta']:+.3f}[{cb1['ci'][0]:+.3f},{cb1['ci'][1]:+.3f}]"
          f" k>=2 {cb2['delta']:+.3f}[{cb2['ci'][0]:+.3f},{cb2['ci'][1]:+.3f}] (n={cb1['n']})", flush=True)
    out["A2"] = a2

    # ---------- A3 knowledge structure / scarcity ----------
    a3 = {}
    # per channel x popularity stratum base rates + rough share; scarcity band 0.2-0.7
    strata = {}
    for ch, cells_key, strat_fn in (
            ("item", "item_llm", lambda c: c["tier"]),
            ("concept", "concept", None),
            ("attribute", "attribute", lambda c: c["atype"])):
        rows = {}
        buckets = collections.defaultdict(list)
        for rec in users:
            for c in rec[cells_key]:
                if c["k"] is None:
                    continue
                if strat_fn is None:
                    # concept: stratify by popularity tercile of member-count
                    key = "all"
                else:
                    key = strat_fn(c)
                buckets[key].append(c["k"])
        for key, ks in buckets.items():
            ks = np.array(ks)
            rate1 = float(np.mean(ks >= 1)); rough = float(np.mean(ks == 1))
            rows[key] = dict(n=len(ks), rate_k1=rate1, rate_k2=float(np.mean(ks == 2)),
                             rough_share=rough, in_scarcity_band=bool(0.2 <= rate1 <= 0.7))
        strata[ch] = rows
    # concept popularity terciles (member-count) for scarcity
    conc_cells = [(c["pop"], c["k"]) for rec in users for c in rec["concept"] if c["k"] is not None]
    if conc_cells:
        pops = np.array([p for p, _ in conc_cells]); ks = np.array([k for _, k in conc_cells])
        q = np.quantile(pops, [1 / 3, 2 / 3])
        for name, lo, hi in (("niche", -1, q[0]), ("mid", q[0], q[1]), ("broad", q[1], 1e18)):
            m = (pops > lo) & (pops <= hi)
            if m.sum():
                rate1 = float(np.mean(ks[m] >= 1))
                strata["concept"][f"tercile_{name}"] = dict(
                    n=int(m.sum()), rate_k1=rate1, rate_k2=float(np.mean(ks[m] == 2)),
                    rough_share=float(np.mean(ks[m] == 1)), in_scarcity_band=bool(0.2 <= rate1 <= 0.7))
    # item within-bank popularity terciles (top-800 is all globally 'famous' -> band it internally)
    item_cells = [(c["cnt"], c["k"]) for rec in users for c in rec["item_llm"] if c["k"] is not None]
    if item_cells:
        cnts = np.array([p for p, _ in item_cells]); ks = np.array([k for _, k in item_cells])
        q = np.quantile(cnts, [1 / 3, 2 / 3])
        for name, lo, hi in (("bank_low", -1, q[0]), ("bank_mid", q[0], q[1]), ("bank_high", q[1], 1e18)):
            m = (cnts > lo) & (cnts <= hi)
            if m.sum():
                rate1 = float(np.mean(ks[m] >= 1))
                strata["item"][f"tercile_{name}"] = dict(
                    n=int(m.sum()), rate_k1=rate1, rate_k2=float(np.mean(ks[m] == 2)),
                    rough_share=float(np.mean(ks[m] == 1)),
                    in_scarcity_band=bool(0.2 <= float(np.mean(ks[m] == 2)) <= 0.7))
    a3["strata"] = strata
    overall_rough = np.mean([1.0 if c["k"] == 1 else 0.0
                             for rec in users for ch in ("concept", "attribute", "item_llm")
                             for c in rec[ch] if c["k"] is not None])
    a3["overall_rough_share"] = float(overall_rough)
    print("  [A3] scarcity strata (answer-rate k>=1 in the 0.2-0.7 habitat):", flush=True)
    for ch in strata:
        for key, r in strata[ch].items():
            flag = "<<HABITAT" if r["in_scarcity_band"] else ""
            print(f"       {ch:9s} {key:14s} n={r['n']:6d} k>=1={r['rate_k1']:.3f} "
                  f"k>=2={r['rate_k2']:.3f} rough={r['rough_share']:.3f} {flag}", flush=True)
    print(f"  [A3] overall rough_idea share = {overall_rough:.3f}", flush=True)
    out["A3"] = a3

    # ---------- verdicts ----------
    def icc_pass(ch):
        r = chan_rows[ch]
        return (r["k1"]["ci"][0] >= 0.05) or (r["k2"]["ci"][0] >= 0.05) or \
               (r["k1"]["icc"] >= 0.05) or (r["k2"]["icc"] >= 0.05)
    a1_pass = {ch: icc_pass(ch) for ch in chan_rows}
    # A2 moderate-band = item mid band (k>=2, where item variance lives) OR concept (k>=1)
    a2_item_mid = a2["item_bands"]["mid"]["taste_near_far_k2"]["ci"][0] > 0
    a2_concept = a2["concept"]["taste_near_far_k1"]["ci"][0] > 0
    a2_pass = bool(a2_item_mid or a2_concept)
    hab = [f"{ch}/{key}" for ch in strata for key, r in strata[ch].items() if r["in_scarcity_band"]]
    out["verdicts"] = dict(A1_icc_pass=a1_pass, A2_item_mid_k2_pass=bool(a2_item_mid),
                           A2_concept_k1_pass=bool(a2_concept), A2_pass=a2_pass, A3_habitat_strata=hab)

    # ---------- write MD ----------
    md_write(f"# Directional Adaptivity Battery -- Answerer v1 (PARTIAL 173-user)\n\n{BANNER}"
             f"Date 2026-07-08. Script `scripts/adaptivity_battery_v1.py`. NO LLM API calls; all local "
             f"compute on cached grids. Environment = the judged answerer-v1 grid merged in-memory "
             f"(163 shard users + 10 grid10 = {len(grid)} raw; {len(users)} pass the fold filter). "
             f"Working merge `{WORKING}` (NOT FROZEN). Paired per-user bootstrap BOOT={BOOT} seed={SEED}. "
             f"Answerability = knowledge != no_clue (k>=1); strong = know_well (k>=2); w_rough={W_ROUGH}.\n\n",
             mode="w")
    md_write("## STAGE A -- FUEL GATES\n\n"
             "**Pre-registered thresholds** (from the original study): A1 ICC>=0.05 beyond log-popularity, "
             "per channel; A2 user-specific-signal validity gap >0 with CI excl 0 in the moderate band; "
             "A3 locate the scarcity habitat (answer-rate in 0.2-0.7).\n\n")
    md_write("### A1 Heterogeneity (one-way random-effects ICC of answerability across users, "
             "log-popularity partialled out)\n\n"
             "| channel | base k>=1 | base k>=2 | ICC(k>=1 \\| logpop) [95% CI] | ICC(k>=2 \\| logpop) [95% CI] | raw ICC k>=1 |\n"
             "|---|--:|--:|---|---|--:|\n")
    for ch in ("concept", "attribute", "item"):
        r = chan_rows[ch]
        md_write(f"| {ch} | {r['base_k1']:.3f} | {r['base_k2']:.3f} | "
                 f"{r['k1']['icc']:.3f}[{r['k1']['ci'][0]:.3f},{r['k1']['ci'][1]:.3f}] | "
                 f"{r['k2']['icc']:.3f}[{r['k2']['ci'][0]:.3f},{r['k2']['ci'][1]:.3f}] | "
                 f"{r['k1_raw']['icc']:.3f} |\n")
    md_write(f"\nA1 verdict (ICC>=0.05 threshold): {a1_pass}. Item channel base rate on LLM-judged "
             "(genuinely uncertain) cells only; rated known-half items are data-filled (all know_well) "
             "and reported separately in A3.\n\n")
    md_write("### A2 Validity gap (user-specific signal)\n\n"
             "The top-800 item bank is entirely global-famous (pr>=0.95), so the design's MODERATE band "
             "is realized as WITHIN-BANK popularity terciles (low/mid/high cnt); mid = moderate. "
             "Non-circular test: within a band, answerability of taste-NEAR items (genre matches the "
             "user's known-liked taste, above per-user median) minus taste-FAR. Item k>=1 is near-ceiling "
             "so the k>=2 (know_well) gap is the informative one for items; concepts span the range.\n\n"
             "| stratum | near-far k>=1 [95% CI] | near-far k>=2 [95% CI] | n users |\n|---|---|---|--:|\n")
    for band in ("low", "mid", "high"):
        b1 = a2["item_bands"][band]["taste_near_far_k1"]; b2 = a2["item_bands"][band]["taste_near_far_k2"]
        md_write(f"| item-{band} | {b1['delta']:+.3f}[{b1['ci'][0]:+.3f},{b1['ci'][1]:+.3f}] | "
                 f"{b2['delta']:+.3f}[{b2['ci'][0]:+.3f},{b2['ci'][1]:+.3f}] | {b1['n']} |\n")
    md_write(f"| concept (all tags) | {cb1['delta']:+.3f}[{cb1['ci'][0]:+.3f},{cb1['ci'][1]:+.3f}] | "
             f"{cb2['delta']:+.3f}[{cb2['ci'][0]:+.3f},{cb2['ci'][1]:+.3f}] | {cb1['n']} |\n")
    md_write(f"\nA2 verdict (user-specific gap>0, CI excl 0): item-mid k>=2 {a2_item_mid}; "
             f"concept k>=1 {a2_concept}; overall A2 pass = {a2_pass}.\n\n")
    md_write("### A3 Knowledge structure / scarcity habitat\n\n"
             f"Overall rough_idea share = {overall_rough:.3f}. Answer-rate in the 0.2-0.7 band = the "
             "adaptivity habitat (scarcity a router can exploit).\n\n"
             "| channel | stratum | n | rate k>=1 | rate k>=2 | rough share | in 0.2-0.7 habitat |\n"
             "|---|---|--:|--:|--:|--:|:--:|\n")
    for ch in strata:
        for key, r in strata[ch].items():
            md_write(f"| {ch} | {key} | {r['n']} | {r['rate_k1']:.3f} | {r['rate_k2']:.3f} | "
                     f"{r['rough_share']:.3f} | {'YES' if r['in_scarcity_band'] else '-'} |\n")
    md_write(f"\nScarcity-habitat strata (adaptivity has purchase here): {hab if hab else 'NONE'}.\n\n")

    json.dump(out, open("experiments/adaptivity_battery_v1_A.json", "w"), indent=1, default=str)
    print(f"\n[stage A] wrote {OUT_MD} (Stage A) + experiments/adaptivity_battery_v1_A.json", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["a", "b", "c"], required=True)
    a = ap.parse_args()
    if a.stage == "a":
        stage_a()
    elif a.stage == "b":
        import battery_stage_b as SB  # noqa
        SB.stage_b()
    elif a.stage == "c":
        import battery_stage_c as SC  # noqa
        SC.stage_c()
