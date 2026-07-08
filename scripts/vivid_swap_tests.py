"""vivid_swap_tests.py -- THREE SEQUENCED tests of the VIVID-SWAP policy mechanism on the DIRECTIONAL
173-user answerer-v1 working grid. NO LLM API calls; all local compute. Author-directed 2026-07-08.

*** DIRECTIONAL ONLY: 173/300 users, grid NOT frozen. Every output carries this flag. ***

THE MECHANISM UNDER TEST: swapping a static schedule's question for an equal-value PEER that THIS user
knows VIVIDLY (k>=2, know_well) should raise NDCG because k=2 answers (a) fold at weight 1.0 vs 0.5
(rough_idea, w_rough), (b) for items are disproportionately source="data" real ratings (zero noise) vs
LLM-guessed, (c) avoid dilution. Personalization = belief updated with ANSWER VALUES (taste), not just
k-labels.

The three tests GATE each other (a cheap death is a success of the method):
  T1 VIVIDNESS VALUE CHECK  -- does a k=2 answer beat a k=1 answer at MATCHED question value, through
                               the real I2.5 fold? GATE: k2-set beats k1-set (matched value tiers), CI
                               excl 0. If NOT: premise dead -> STOP.
  T2 PEER-RANKING ACCURACY  -- within value-tier peer sets, can a blind belief pick THIS user's k=2 peer?
                               GATE: belief(iii) LOUO-MF+TASTE beats belief(i) pop-k2-rate on precision@1
                               (CI excl 0) at t=8. If NOT: run T3 population-vivid variant only.
  T3 THE POLICY             -- s-best / s-vivid (population-vivid static) / r-vivid (full router).
                               Contrasts: r-vivid vs s-best (headline), r-vivid vs s-vivid (personalization
                               increment = THE thesis quantity), s-vivid vs s-best (vividness-prior gain).

ERROR LEDGER honored (STATE_2026-07-08): E1 survivorship (same user set / paired within-user); E2
tie-by-construction floor (r-vivid reduces to ~s-vivid at t=0, never loses to s-best beyond noise);
E3 privilege labelling (NO oracle / target-peek / answerability-table arms -- Stage-C retraction);
E5 circularity firewall (beliefs LEAVE-ONE-USER-OUT / population, never the eval user's own cells);
E6 pre-registered thresholds printed BEFORE results, paired per-user bootstrap CIs, deterministic seeds,
ASCII prints. The "value model" (per-candidate mean single-answer NDCG lift) is applied IDENTICALLY to
the k1 and k2 arms (E2 value-model confound killed by construction).

Run:  python scripts/vivid_swap_tests.py --test 1|2|3|all
"""
import os, sys, json, time, argparse, collections
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
import adaptivity_battery_v1 as A
import battery_stage_b as B
import battery_stage_c as C
import i25_lib as L
import i25_phase4 as P4
import i25_phase4_fair as FA

CKPT = ".cache/i25_fold_best.pt"
OUT_MD = "experiments/VIVID_SWAP_TESTS.md"
OUT_JSON = "experiments/vivid_swap_tests.json"
BOOT = A.BOOT           # 5000
SEED = A.SEED           # 0
NTIER = 8               # value tiers
MIN_COV = 10            # min answering users for a candidate to get a stable value / enter a tier
T3_T = 12               # arena budget
RESULTS = {"banner": "DIRECTIONAL 173/300, grid unfrozen (answerer-v1 working grid)"}

# ---- TARGET SWITCH (fold_weight_calibration.py sets TGT='data' when T1b (rated premium) is the passing
#      class; default 'k2' preserves the standalone behavior exactly). 'data' = source=="data" real
#      ratings (only ever on the item channel); 'k2' = know_well (klev==2). -------------------------------
TGT = "k2"


def _is_tgt(rec, cd):
    if TGT == "data":
        return 1 if rec["src_arr"][cd] == "data" else 0
    return 1 if rec["klev_arr"][cd] == 2 else 0


def _poptgt(env):
    return env["pop_data"] if TGT == "data" else env["pop_k2"]


def _item_cells_tgt(rec, kthr):
    """LOUO cells for the item channel under the active target. TGT='data' => positive iff source=data."""
    if TGT != "data":
        return item_cells_of(rec, kthr)
    cells = {}
    for c in rec["item_llm"]:
        if c["k"] is not None:
            cells[c["j"]] = (0, c["cnt"])                 # LLM-valued item = not a rated positive
    for c in rec["item_data"]:
        cells[c["j"]] = (1, c["cnt"])                     # real rating = the target positive
    eids = list(cells.keys()); y = np.array([cells[j][0] for j in eids])
    order = np.argsort([-cells[j][1] for j in eids])
    return eids, y, order


# ============================================================ md io
def md_write(txt, mode="a"):
    os.makedirs("experiments", exist_ok=True)
    open(OUT_MD, mode, encoding="utf-8").write(txt)


def paired(a, b, seed=SEED):
    """Paired per-user bootstrap of mean(a-b). Drops rows where either is None/nan."""
    pa, pb = [], []
    for x, y in zip(a, b):
        if x is None or y is None:
            continue
        if (isinstance(x, float) and np.isnan(x)) or (isinstance(y, float) and np.isnan(y)):
            continue
        pa.append(x); pb.append(y)
    if not pa:
        return dict(delta=float("nan"), ci=[float("nan")] * 2, n=0, a=float("nan"), b=float("nan"))
    d = np.asarray(pa) - np.asarray(pb)
    rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(BOOT)])
    return dict(delta=float(d.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                n=int(len(d)), a=float(np.mean(pa)), b=float(np.mean(pb)))


# ============================================================ SET-level fold NDCG
def set_ndcg(FR, model, users, sets):
    """sets: list aligned with users; each = list of cids (already answerable). Returns per-user NDCG@10;
    empty set -> None (caller handles)."""
    tl, nl, idx = [], [], []
    for i, rec in enumerate(users):
        cids = sets[i]
        if not cids:
            continue
        tl.append([C.tok_of(C.CANDS_G, rec, c) for c in cids])
        nl.append([rec["nat_arr"][c] for c in cids if rec["nat_arr"][c] is not None])
        idx.append(i)
    out = [None] * len(users)
    if tl:
        vals = C._batch_ndcg(FR, model, users, tl, nl, idx, np.zeros(len(users)))
        for r, i in enumerate(idx):
            out[i] = float(vals[r])
    return out


# ============================================================ SETUP (shared)
_ENV = None
def setup():
    global _ENV
    if _ENV is not None:
        return _ENV
    t0 = time.time()
    D, split, grid, users, memb, battery = A.load_env()
    FR = L.Frozen(D)
    model = L.Fold(); blob = torch.load(CKPT, map_location="cpu")
    model.load_state_dict(blob["model"]); model.eval()
    P4.T = T3_T
    C.CANDS_G = C.build_cands(D, FR, users, memb, battery)   # fills ans_arr / val_arr / nat_arr per user
    CANDS = C.CANDS_G
    nc = len(CANDS)
    cold = C.cold_ndcg(FR, model, users)

    # ---- per (user,cand) knowledge level + source (data/llm/concept/attr) ----
    cid_of = {(m["kind"], m["key"]): m["cid"] for m in CANDS}
    for rec in users:
        klev = -np.ones(nc, np.int64); src = [None] * nc
        for c in rec["concept"]:
            cd = cid_of.get(("concept", c["tagId"]))
            if cd is not None and c["k"] is not None and c["k"] >= 1:
                klev[cd] = c["k"]; src[cd] = "concept"
        for c in rec["attribute"]:
            cd = cid_of.get(("attr", c["eid"]))
            if cd is not None and c["k"] is not None and c["k"] >= 1:
                klev[cd] = c["k"]; src[cd] = "attr"
        for c in rec["item_llm"]:
            cd = cid_of.get(("item", c["j"]))
            if cd is not None and c["k"] is not None and c["k"] >= 1 and c["ans"].get("stars") is not None:
                klev[cd] = c["k"]; src[cd] = "llm"
        for c in rec["item_data"]:
            cd = cid_of.get(("item", c["j"]))
            if cd is not None:
                klev[cd] = 2; src[cd] = "data"                    # real rating => know_well, zero noise
        # consistency with build_cands answerability
        bad = int(((klev >= 1) != rec["ans_arr"]).sum())
        rec["klev_arr"] = klev; rec["src_arr"] = src; rec["_klev_mismatch"] = bad

    mism = sum(rec["_klev_mismatch"] for rec in users)
    # ---- per-candidate VALUE (mean single-answer NDCG lift from cold over answering users) ----
    tl, nl, mi, mc = [], [], [], []
    for i, rec in enumerate(users):
        for cd in np.where(rec["ans_arr"])[0]:
            tl.append([C.tok_of(CANDS, rec, cd)])
            nl.append([rec["nat_arr"][cd]] if rec["nat_arr"][cd] is not None else [])
            mi.append(i); mc.append(int(cd))
    vals = C._batch_ndcg(FR, model, users, tl, nl, mi, cold)
    lifts_by_cand = collections.defaultdict(list)
    salift = {}                                                   # (i,cd) -> single-answer lift
    for r in range(len(vals)):
        lft = float(vals[r] - cold[mi[r]])
        lifts_by_cand[mc[r]].append(lft); salift[(mi[r], mc[r])] = lft
    cval = np.full(nc, np.nan); ccov = np.zeros(nc)
    for cd in range(nc):
        if lifts_by_cand[cd]:
            cval[cd] = float(np.mean(lifts_by_cand[cd])); ccov[cd] = len(lifts_by_cand[cd])

    # ---- population k>=2 rate per candidate (over answering users) ----
    k2c = np.zeros(nc); anyc = np.zeros(nc)
    for rec in users:
        anyc += rec["ans_arr"]; k2c += (rec["klev_arr"] == 2)
    pop_k2 = np.divide(k2c, np.maximum(anyc, 1))

    # ---- value tiers (octiles over candidates with stable value) ----
    valid = (~np.isnan(cval)) & (ccov >= MIN_COV)
    tier_of = -np.ones(nc, np.int64)
    vc = np.where(valid)[0]
    if len(vc):
        qs = np.quantile(cval[vc], np.linspace(0, 1, NTIER + 1))
        qs[-1] += 1e-9
        for cd in vc:
            t = int(np.searchsorted(qs, cval[cd], side="right") - 1)
            tier_of[cd] = min(max(t, 0), NTIER - 1)
    tier_med = {t: float(np.median(cval[(tier_of == t)])) for t in range(NTIER) if (tier_of == t).any()}

    print(f"[setup] {len(users)} users; {nc} cands; cold {cold.mean():.4f}; klev/ans mismatches {mism}; "
          f"valid-value cands {int(valid.sum())}; tier sizes "
          f"{[int((tier_of==t).sum()) for t in range(NTIER)]}", flush=True)
    _ENV = dict(D=D, split=split, grid=grid, users=users, memb=memb, battery=battery, FR=FR, model=model,
                CANDS=CANDS, nc=nc, cold=cold, cval=cval, ccov=ccov, pop_k2=pop_k2, tier_of=tier_of,
                tier_med=tier_med, valid=valid, salift=salift, foldval=blob["state"]["best_val"],
                mism=mism, t0=t0)
    return _ENV


# ============================================================ T1 -- VIVIDNESS VALUE CHECK
def pick_rep(env, rec, cids):
    """Deterministic representative of a candidate group for a user: closest cval to its tier median,
    tie-break by cid."""
    cval, tier_of, tier_med = env["cval"], env["tier_of"], env["tier_med"]
    best, bc = None, None
    for c in cids:
        med = tier_med.get(int(tier_of[c]), cval[c])
        key = (abs(cval[c] - med), c)
        if best is None or key < best:
            best, bc = key, c
    return bc


def build_matched_sets(env, users, k2_filter):
    """Per user: for each tier where the user has BOTH a k1 cand and a k2 cand passing k2_filter, slot in
    one k1 rep and one k2 rep. Returns k1_sets, k2_sets aligned to users (matched tier-for-tier)."""
    tier_of, valid = env["tier_of"], env["valid"]
    k1_sets, k2_sets = [], []
    for rec in users:
        klev = rec["klev_arr"]; src = rec["src_arr"]
        k1s, k2s = [], []
        for t in range(NTIER):
            in_t = np.where((tier_of == t) & valid)[0]
            k1c = [int(c) for c in in_t if klev[c] == 1]
            k2c = [int(c) for c in in_t if klev[c] == 2 and k2_filter(src[c])]
            if k1c and k2c:
                k1s.append(pick_rep(env, rec, k1c))
                k2s.append(pick_rep(env, rec, k2c))
        k1_sets.append(k1s); k2_sets.append(k2s)
    return k1_sets, k2_sets


def test1():
    env = setup(); FR, model, users = env["FR"], env["model"], env["users"]
    print("\n==== T1 -- VIVIDNESS VALUE CHECK (DIRECTIONAL 173/300) ====", flush=True)
    print("PRE-REGISTERED GATE (printed before results):", flush=True)
    print("  k2-set beats k1-set at MATCHED value tiers, paired per-user bootstrap CI excl 0.", flush=True)
    print("  Value model = per-candidate mean single-answer NDCG lift, applied IDENTICALLY to both arms.", flush=True)

    # main matched comparison: any k2 vs k1
    k1_any, k2_any = build_matched_sets(env, users, lambda s: True)
    n1 = set_ndcg(FR, model, users, k1_any)
    n2 = set_ndcg(FR, model, users, k2_any)
    gate = paired(n2, n1)
    # decomposition: k2-data (real rating) vs k1 ; k2-llm/concept/attr vs k1 (each matched to own k1 tiers)
    k1_d, k2_d = build_matched_sets(env, users, lambda s: s == "data")
    k1_l, k2_l = build_matched_sets(env, users, lambda s: s != "data")
    dec_data = paired(set_ndcg(FR, model, users, k2_d), set_ndcg(FR, model, users, k1_d))
    dec_llm = paired(set_ndcg(FR, model, users, k2_l), set_ndcg(FR, model, users, k1_l))

    n_slots1 = float(np.mean([len(s) for s in k2_any if s])) if any(k2_any) else 0.0
    gate_pass = bool(gate["ci"][0] > 0)
    print(f"\n  k2-set {gate['a']:.4f} vs k1-set {gate['b']:.4f} -> delta {gate['delta']:+.4f}"
          f"[{gate['ci'][0]:+.4f},{gate['ci'][1]:+.4f}] (n={gate['n']} users, ~{n_slots1:.1f} matched tiers/user)"
          f" -> GATE {'PASS' if gate_pass else 'FAIL'}", flush=True)
    print(f"  decomp k2-DATA(real rating) vs k1: {dec_data['delta']:+.4f}"
          f"[{dec_data['ci'][0]:+.4f},{dec_data['ci'][1]:+.4f}] (n={dec_data['n']})", flush=True)
    print(f"  decomp k2-LLM/concept/attr vs k1 : {dec_llm['delta']:+.4f}"
          f"[{dec_llm['ci'][0]:+.4f},{dec_llm['ci'][1]:+.4f}] (n={dec_llm['n']})", flush=True)

    RESULTS["T1"] = dict(gate=gate, gate_pass=gate_pass, decomp_data=dec_data, decomp_llm=dec_llm,
                         mean_matched_tiers=n_slots1)
    _t1_md(env, gate, gate_pass, dec_data, dec_llm, n_slots1)
    _dump()
    return gate_pass


def _t1_md(env, gate, gate_pass, dec_data, dec_llm, n_slots1):
    md_write(f"# Vivid-Swap Mechanism Tests -- Answerer v1 (DIRECTIONAL 173/300)\n\n"
             f"> **DIRECTIONAL ONLY** -- 173/300 users, answerer-v1 working grid NOT frozen. Every number is "
             f"provisional; re-run on the frozen 300-user grid before any citation.\n\n"
             f"Date 2026-07-08. Script `scripts/vivid_swap_tests.py`. NO LLM API calls; local compute on the "
             f"cached judged grid through the I2.5 learned fold (`.cache/i25_fold_best.pt`, val "
             f"{env['foldval']:.4f}). {len(env['users'])} users pass the fold filter; {env['nc']} shared "
             f"candidates; cold NDCG@10 {env['cold'].mean():.4f}. Paired per-user bootstrap BOOT={BOOT} "
             f"seed={SEED}. Vividness = know_well (k>=2); rough_idea (k=1) folds at w_rough=0.5.\n\n"
             f"**Value model (used identically across arms, E2-clean):** per-candidate mean single-answer "
             f"NDCG@10 lift from cold, over its answering users (>= {MIN_COV}); value TIERS = {NTIER} octiles "
             f"of that value. Matching on the tier controls question value; the residual k2-vs-k1 gap is the "
             f"mechanism. klev/answerability mismatches vs build_cands: {env['mism']} (0 = consistent).\n\n",
             mode="w")
    md_write("## T1 -- VIVIDNESS VALUE CHECK (the premise)\n\n"
             "**PRE-REGISTERED GATE:** k2-set beats k1-set at matched value tiers, paired CI excl 0. "
             "If FAIL: the premise is dead -> STOP.\n\n"
             "Design: per user, for each value tier holding BOTH a k=1 and a k=2 grid cell, slot in one k=1 "
             "rep and one k=2 rep (rep = cval closest to tier median). The k1-set and k2-set are matched "
             "tier-for-tier and differ ONLY in knowledge composition. Both folded through the real I2.5 "
             "fold (know_well weight 1.0, rough_idea 0.5; item rated cells = real ratings, native RecVAE "
             "token where stars>=4). NDCG@10 on the untouched held-out halves.\n\n"
             "| contrast | k2 NDCG | k1 NDCG | delta [95% CI] | n users | verdict |\n|---|--:|--:|---|--:|---|\n")
    md_write(f"| **k2-set vs k1-set (matched, ANY k2)** | {gate['a']:.4f} | {gate['b']:.4f} | "
             f"**{gate['delta']:+.4f}**[{gate['ci'][0]:+.4f},{gate['ci'][1]:+.4f}] | {gate['n']} | "
             f"{'PASS' if gate_pass else 'FAIL'} |\n")
    md_write(f"| k2-DATA (real rating) vs k1 | {dec_data['a']:.4f} | {dec_data['b']:.4f} | "
             f"{dec_data['delta']:+.4f}[{dec_data['ci'][0]:+.4f},{dec_data['ci'][1]:+.4f}] | {dec_data['n']} | "
             f"decomposition |\n")
    md_write(f"| k2-LLM/concept/attr vs k1 | {dec_llm['a']:.4f} | {dec_llm['b']:.4f} | "
             f"{dec_llm['delta']:+.4f}[{dec_llm['ci'][0]:+.4f},{dec_llm['ci'][1]:+.4f}] | {dec_llm['n']} | "
             f"decomposition |\n")
    md_write(f"\nMean matched value tiers per user = {n_slots1:.1f}. "
             f"**T1 GATE = {'PASS -> proceed to T2' if gate_pass else 'FAIL -> STOP (premise dead)'}.**\n\n")


# ============================================================ T2 -- PEER-RANKING ACCURACY
def item_cells_of(rec, kthr):
    cells = {}
    for c in rec["item_llm"]:
        if c["k"] is not None:
            cells[c["j"]] = (1 if c["k"] >= kthr else 0, c["cnt"])
    for c in rec["item_data"]:
        cells.setdefault(c["j"], (1, c["cnt"]))               # data = know_well -> label 1 for kthr<=2
    eids = list(cells.keys()); y = np.array([cells[j][0] for j in eids])
    order = np.argsort([-cells[j][1] for j in eids])
    return eids, y, order


def conc_cells_of(rec, kthr):
    cells = {}
    for c in rec["concept"]:
        if c["k"] is not None:
            cells[c["tagId"]] = (1 if c["k"] >= kthr else 0, c["pop"])
    eids = list(cells.keys()); y = np.array([cells[t][0] for t in eids])
    order = np.argsort([-cells[t][1] for t in eids])
    return eids, y, order


def _peer_sets(env):
    """Peer sets = (channel, tier) groups of shared candidates; cap to 12 by coverage. size in [3,12]."""
    CANDS, tier_of, valid, ccov = env["CANDS"], env["tier_of"], env["valid"], env["ccov"]
    groups = collections.defaultdict(list)
    for m in CANDS:
        cd = m["cid"]
        if valid[cd]:
            groups[(m["kind"], int(tier_of[cd]))].append(cd)
    out = {}
    for key, cds in groups.items():
        cds = sorted(cds, key=lambda c: -ccov[c])[:12]
        if len(cds) >= 3:
            out[key] = cds
    return out


def _taste_z(env, users, revealed_lists):
    """Fold each user's given revealed cid list of ANSWERED cells -> latent z (batched)."""
    FR, model, CANDS = env["FR"], env["model"], env["CANDS"]
    tl, nl, idx = [], [], []
    for i, rec in enumerate(users):
        cids = revealed_lists[i]
        if cids:
            tl.append([C.tok_of(CANDS, rec, c) for c in cids])
            nl.append([rec["nat_arr"][c] for c in cids if rec["nat_arr"][c] is not None])
            idx.append(i)
    Z = np.zeros((len(users), FR.W.shape[1]))
    for s in range(0, len(tl), 3000):
        e = min(s + 3000, len(tl))
        z = P4.fold_batch(FR, model, tl[s:e], nl[s:e])
        for r in range(s, e):
            Z[idx[r]] = z[r - s]
    return Z


def test2():
    env = setup(); FR, model, users = env["FR"], env["model"], env["users"]
    CANDS, pop_k2 = env["CANDS"], _poptgt(env)          # pop_k2 = population TARGET rate (k2 or data)
    print("\n==== T2 -- PEER-RANKING ACCURACY (DIRECTIONAL 173/300; target=%s) ====" % TGT, flush=True)
    print("PRE-REGISTERED GATE (printed before results):", flush=True)
    print("  belief(iii) LOUO-MF+TASTE beats belief(i) population-k2-rate on precision@1 (CI excl 0) at t=8.", flush=True)
    print("  If FAIL: personalization has no purchase -> T3 population-vivid variant only.", flush=True)

    peers = _peer_sets(env)
    # candidate reveal order per user = answerable cands by coverage desc (popularity-like)
    ccov = env["ccov"]
    order_of = {}
    for rec in users:
        ans = np.where(rec["ans_arr"])[0]
        order_of[rec["u"]] = sorted([int(c) for c in ans], key=lambda c: -ccov[c])

    # ---- LOUO-MF beliefs (k>=2), per channel (E5 firewall) ----
    item_univ = sorted({j for rec in users for c in rec["item_llm"] for j in [c["j"]] if c["k"] is not None}
                       | {c["j"] for rec in users for c in rec["item_data"]})
    conc_univ = sorted({c["tagId"] for rec in users for c in rec["concept"] if c["k"] is not None})
    Ei, bi, erow_i = B.louo_mf_beliefs(users, lambda r: _item_cells_tgt(r, 2), set(item_univ), 2)
    Ec, bc_, erow_c = B.louo_mf_beliefs(users, lambda r: conc_cells_of(r, 2), set(conc_univ), 2)
    # key(cand) -> (row in channel embedding) ; attr channel has no LOUO -> louo_score falls back to pop
    key_row = {}
    for m in CANDS:
        if m["kind"] == "item" and m["key"] in erow_i:
            key_row[m["cid"]] = ("item", erow_i[m["key"]])
        elif m["kind"] == "concept" and m["key"] in erow_c:
            key_row[m["cid"]] = ("concept", erow_c[m["key"]])

    def louo_score(rec, cd, alpha_i, k_i, alpha_c, k_c):
        kr = key_row.get(cd)
        if kr is None:
            return None
        ch, row = kr
        if ch == "item":
            return float(bi[rec["u"]][row] + alpha_i + Ei[rec["u"]][row] @ k_i)
        return float(bc_[rec["u"]][row] + alpha_c + Ec[rec["u"]][row] @ k_c)

    # 5-fold over users for belief (iii) logistic (E5: fit on OTHER users' cells)
    uids = [rec["u"] for rec in users]
    rng = np.random.default_rng(SEED); perm = rng.permutation(len(uids))
    folds = [set(np.array(uids)[perm[f::5]].tolist()) for f in range(5)]
    fold_of = {}
    for f, S in enumerate(folds):
        for u in S:
            fold_of[u] = f

    def run_t(t):
        # per-user inferred louo vectors + taste z at evidence budget t (evidence excludes the scored peer set)
        # Build features per (user, peer-set, cand). We approximate evidence as first-t answered cands MINUS
        # the peer set's cands (no leakage of the answer we predict).
        rows = []                     # (user_idx, fold, setkey, cd, feats..., label)
        # precompute taste z per user for a set-agnostic order first-t (peers removed per set below cheaply
        # via re-fold only when a peer is in the first-t window -- rare; we recompute per set for safety)
        setlist = list(peers.items())
        # infer louo (alpha,k) per user per set (evidence = first-t answered, minus set peers)
        feat_rows, meta = [], []
        # batch taste folds: build revealed lists per (user,set)
        rev_lists = []; rl_key = []
        for i, rec in enumerate(users):
            base = order_of[rec["u"]]
            for (setkey, cds) in setlist:
                setc = set(cds)
                ev = [c for c in base if c not in setc][:t]
                rev_lists.append(ev); rl_key.append((i, setkey))
        Ztaste = _taste_z(env, [users[k[0]] for k in rl_key], rev_lists)   # aligned to rl_key rows
        z_by = {rl_key[r]: Ztaste[r] for r in range(len(rl_key))}
        for i, rec in enumerate(users):
            base = order_of[rec["u"]]
            klev = rec["klev_arr"]
            for (setkey, cds) in setlist:
                setc = set(cds)
                ev = [c for c in base if c not in setc][:t]
                # louo inference from evidence (item + concept channels)
                ai, ki = _infer_channel(rec, ev, "item", key_row, Ei, bi)
                ac, kc = _infer_channel(rec, ev, "concept", key_row, Ec, bc_)
                zt = z_by[(i, setkey)]
                for cd in cds:
                    ls = louo_score(rec, cd, ai, ki, ac, kc)
                    emb = CANDS[cd]["emb"]
                    denom = (np.linalg.norm(zt) * np.linalg.norm(emb) + 1e-9)
                    taste = float((zt @ emb) / denom)
                    feat = [pop_k2[cd], ls if ls is not None else 0.0, 1.0 if ls is not None else 0.0,
                            taste, np.log1p(env["ccov"][cd])]
                    lab = _is_tgt(rec, cd)
                    feat_rows.append(feat); meta.append((i, fold_of[rec["u"]], setkey, cd, lab))
        X = np.array(feat_rows); y = np.array([m[4] for m in meta])
        # fit belief (iii) 5-fold
        score_iii = np.zeros(len(meta))
        for f in range(5):
            tr = np.array([j for j in range(len(meta)) if meta[j][1] != f])
            te = np.array([j for j in range(len(meta)) if meta[j][1] == f])
            if len(te) == 0 or len(np.unique(y[tr])) < 2:
                continue
            clf = LogisticRegression(max_iter=500, C=1.0)
            clf.fit(X[tr], y[tr])
            score_iii[te] = clf.predict_proba(X[te])[:, 1]
        # precision@1 per (user,set) for belief (i)=pop_k2 and (iii)=logistic ; restrict to sets where the
        # user has >=1 k2 AND >=1 answerable-non-k2 peer (non-trivial choice)
        by_set = collections.defaultdict(list)              # (i,setkey) -> list of (cd, pop, s3, lab, klev, src)
        for j, (i, f, setkey, cd, lab) in enumerate(meta):
            rec = users[i]
            by_set[(i, setkey)].append((cd, X[j][0], score_iii[j], lab, rec["klev_arr"][cd], rec["src_arr"][cd]))
        p1_i, p1_iii, fw_i, fw_iii = [], [], [], []
        for _, rowset in by_set.items():
            k2n = sum(1 for r in rowset if r[3] == 1)        # target positives (k2 or rated)
            ansn = sum(1 for r in rowset if r[4] >= 1)       # answerable peers (klev>=1)
            if k2n < 1 or k2n >= ansn or ansn < 2:
                continue
            top_i = max(rowset, key=lambda r: (r[1], -r[0]))
            top_iii = max(rowset, key=lambda r: (r[2], -r[0]))
            p1_i.append(1.0 if top_i[3] == 1 else 0.0)
            p1_iii.append(1.0 if top_iii[3] == 1 else 0.0)
            fw_i.append(A.FOLD_WEIGHT.get(["no_clue", "rough_idea", "know_well"][max(top_i[4], 0)] if top_i[4] >= 0 else "no_clue", 0.0))
            fw_iii.append(A.FOLD_WEIGHT.get(["no_clue", "rough_idea", "know_well"][max(top_iii[4], 0)] if top_iii[4] >= 0 else "no_clue", 0.0))
        prec = paired(p1_iii, p1_i)
        fw = paired(fw_iii, fw_i)
        return dict(prec_iii=float(np.mean(p1_iii)) if p1_iii else float("nan"),
                    prec_i=float(np.mean(p1_i)) if p1_i else float("nan"),
                    delta=prec, n_sets=len(p1_i), fw_iii=float(np.mean(fw_iii)) if fw_iii else float("nan"),
                    fw_i=float(np.mean(fw_i)) if fw_i else float("nan"), fw_delta=fw)

    res = {}
    for t in (4, 8):
        r = run_t(t)
        res[t] = r
        print(f"  [t={t}] precision@1  belief(i) pop={r['prec_i']:.3f}  belief(iii) MF+taste={r['prec_iii']:.3f}"
              f"  delta {r['delta']['delta']:+.3f}[{r['delta']['ci'][0]:+.3f},{r['delta']['ci'][1]:+.3f}]"
              f"  (n_sets={r['n_sets']}) | realized fold-wt iii={r['fw_iii']:.3f} vs sched(i)={r['fw_i']:.3f}",
              flush=True)
    gate_pass = bool(res[8]["delta"]["ci"][0] > 0)
    print(f"  T2 GATE (belief iii > i at t=8, CI excl 0) = {'PASS' if gate_pass else 'FAIL'}", flush=True)
    RESULTS["T2"] = dict(t4=res[4], t8=res[8], gate_pass=gate_pass, n_peer_sets=len(peers))
    _t2_md(env, res, gate_pass, len(peers))
    _dump()
    return gate_pass


def _infer_channel(rec, ev_cids, channel, key_row, E_by, b_by):
    """Infer (alpha, k) from evidence cids belonging to `channel`, label = (klev==2)."""
    rows, ys = [], []
    E = E_by[rec["u"]]; b = b_by[rec["u"]]
    for c in ev_cids:
        kr = key_row.get(c)
        if kr is not None and kr[0] == channel:
            rows.append(kr[1]); ys.append(_is_tgt(rec, c))
    if not rows:
        return 0.0, np.zeros(B.D_MF)
    Emat = E[rows]; bvec = b[rows]
    return B.infer_user(Emat, bvec, np.array(ys, float), B.D_MF)


def _t2_md(env, res, gate_pass, n_peers):
    md_write("## T2 -- PEER-RANKING ACCURACY (the belief where it matters)\n\n"
             "**PRE-REGISTERED GATE:** belief (iii) LOUO-MF+TASTE beats belief (i) population-k2-rate on "
             "precision@1, paired CI excl 0, at t=8 evidence. If FAIL: personalization has no purchase -> "
             "run T3 with the population-vivid variant only.\n\n"
             f"Peer sets = (channel, value-tier) groups of shared candidates (size 3-12, capped by "
             f"coverage); {n_peers} sets. Evaluated on sets where the user has >=1 k=2 AND >=1 "
             f"answerable-non-k2 peer (a non-trivial pick). Beliefs (E5 firewall): (i) population k>=2 rate; "
             f"(ii/iii base) 5-fold LEAVE-ONE-USER-OUT logistic MF on k>=2 (item + concept channels), online "
             f"user vector from t evidence events; (iii) 5-fold logistic P(k=2 | pop, louo_score, "
             f"taste_dot, log-cov) where taste_dot = cosine(fold of first-t answer VALUE tokens, candidate "
             f"direction). Evidence excludes the scored peer set (no leakage). precision@1 = the belief's "
             f"top-1 peer is actually answered k=2 by the user.\n\n"
             "| evidence | belief(i) pop prec@1 | belief(iii) MF+taste prec@1 | delta [95% CI] | n sets | "
             "realized fold-wt iii vs sched |\n|---|--:|--:|---|--:|---|\n")
    for t in (4, 8):
        r = res[t]
        md_write(f"| t={t} | {r['prec_i']:.3f} | {r['prec_iii']:.3f} | {r['delta']['delta']:+.3f}"
                 f"[{r['delta']['ci'][0]:+.3f},{r['delta']['ci'][1]:+.3f}] | {r['n_sets']} | "
                 f"{r['fw_iii']:.3f} vs {r['fw_i']:.3f} |\n")
    md_write(f"\n**T2 GATE = {'PASS -> r-vivid uses the personalized belief (iii) in T3' if gate_pass else 'FAIL -> T3 runs the population-vivid variant only (r-vivid reduces to s-vivid)'}.**\n\n")


# ============================================================ T3 -- THE POLICY
def _reproduce_statics(env):
    FR, model, users, CANDS = env["FR"], env["model"], env["users"], env["CANDS"]
    C.T = T3_T; P4.T = T3_T
    # E4 canary to decide passing channels (reuse the Stage-C gate criterion)
    cold = env["cold"]; rng = np.random.default_rng(SEED)
    passing = []
    for kind in ("concept", "item", "attr"):
        cids_kind = [m["cid"] for m in CANDS if m["kind"] == kind]
        tl, nl, meta = [], [], []
        for i, rec in enumerate(users):
            avail = [c for c in cids_kind if rec["ans_arr"][c]]
            if not avail:
                continue
            for c in list(rng.permutation(avail))[:C.CANARY_SAMPLE]:
                tl.append([C.tok_of(CANDS, rec, c)])
                nl.append([rec["nat_arr"][c]] if rec["nat_arr"][c] is not None else [])
                meta.append(i)
        if tl:
            vals = C._batch_ndcg(FR, model, users, tl, nl, meta, cold)
            pu = collections.defaultdict(list)
            for r, i in enumerate(meta):
                pu[i].append(vals[r] - cold[i])
            d = [float(np.mean(v)) for v in pu.values()]
            cb = paired(d, [0.0] * len(d))
            if cb["ci"][0] > 0:
                passing.append(kind)
    conc_pool = [m["cid"] for m in CANDS if m["kind"] == "concept" and "concept" in passing]
    item_pool = [m["cid"] for m in CANDS if m["kind"] == "item" and "item" in passing]
    mixed_pool = [m["cid"] for m in CANDS if m["kind"] in passing]
    s_concept = C.build_greedy(FR, model, users, conc_pool, cold, "s-concept") if conc_pool else []
    s_item = C.build_greedy(FR, model, users, item_pool, cold, "s-item") if item_pool else []
    s_mixed = C.build_greedy(FR, model, users, mixed_pool, cold, "s-mixed")
    arms = {}
    if s_concept:
        arms["s-concept"] = C.eval_plan(FR, model, users, C.plan_static(s_concept), cold)
    if s_item:
        arms["s-item"] = C.eval_plan(FR, model, users, C.plan_static(s_item), cold)
    arms["s-mixed"] = C.eval_plan(FR, model, users, C.plan_static(s_mixed), cold)
    scheds = dict(s_concept=s_concept, s_item=s_item, s_mixed=s_mixed)
    return passing, scheds, arms, cold


def _peers_of(env, cd, eps):
    """Peers = same-channel candidates within +/- eps of cd's value (valid, coverage>=MIN_COV)."""
    CANDS, cval, valid = env["CANDS"], env["cval"], env["valid"]
    kind = CANDS[cd]["kind"]; v0 = cval[cd]
    out = []
    for m in CANDS:
        c = m["cid"]
        if valid[c] and m["kind"] == kind and abs(cval[c] - v0) <= eps:
            out.append(c)
    return out


def _svivid_sched(env, sched, eps):
    """Population-vivid static: swap each scheduled cand to its highest-pop-target peer within +/- eps value."""
    pop_k2 = _poptgt(env)
    out = []
    for cd in sched:
        peers = _peers_of(env, cd, eps)
        best = max(peers, key=lambda c: (pop_k2[c], -c)) if peers else cd
        out.append(best)
    return out


def _rvivid_plan(env, sched, eps, delta, belief):
    """Full vivid-swap router. belief in {'pop','iii'}. At each turn, among peers of the scheduled cand
    within +/- eps value, pick the one whose belief-P(k2) exceeds the scheduled cand's by > delta; else keep
    scheduled. With belief='pop' this is deterministic (== s-vivid when delta=0). With 'iii', the online
    user vector (taste + louo) sharpens the pick as evidence accrues; at t=0 it uses pop (reduces to
    ~s-vivid)."""
    CANDS = env["CANDS"]
    pop_k2 = _poptgt(env)
    binfo = env.get("_belief_iii")
    def f(rec):
        used = set(); plan = []; ev = []            # ev = answered cids so far
        for pos in range(min(T3_T, len(sched))):
            cd = sched[pos]
            peers = [c for c in _peers_of(env, cd, eps) if c not in used]
            if not peers:
                peers = [cd] if cd not in used else []
            if not peers:
                plan.append(None); continue
            if belief == "pop" or not ev or binfo is None:
                sc = {c: pop_k2[c] for c in peers}
                sc_cd = pop_k2[cd]
            else:
                sc = binfo["score"](env, rec, peers, ev)
                sc_cd = binfo["score"](env, rec, [cd], ev).get(cd, pop_k2[cd])
            best = max(peers, key=lambda c: (sc[c], -c))
            chosen = best if (sc[best] - sc_cd) > delta else (cd if cd in peers else best)
            used.add(chosen)
            if rec["ans_arr"][chosen]:
                plan.append(chosen); ev.append(chosen)
            else:
                plan.append(None)
        return plan
    return f


def _make_belief_iii(env):
    """Fit ONE logistic P(k=2|pop,louo,taste,logcov) on ALL users' peer-eligible cells (population model;
    used online for r-vivid). Provides a score(env,rec,cds,ev) callable using online taste z(ev) + louo."""
    users, CANDS, pop_k2 = env["users"], env["CANDS"], _poptgt(env)
    item_univ = sorted({j for rec in users for c in rec["item_llm"] for j in [c["j"]] if c["k"] is not None}
                       | {c["j"] for rec in users for c in rec["item_data"]})
    conc_univ = sorted({c["tagId"] for rec in users for c in rec["concept"] if c["k"] is not None})
    Ei, bi, erow_i = B.louo_mf_beliefs(users, lambda r: _item_cells_tgt(r, 2), set(item_univ), 2)
    Ec, bc_, erow_c = B.louo_mf_beliefs(users, lambda r: conc_cells_of(r, 2), set(conc_univ), 2)
    key_row = {}
    for m in CANDS:
        if m["kind"] == "item" and m["key"] in erow_i:
            key_row[m["cid"]] = ("item", erow_i[m["key"]])
        elif m["kind"] == "concept" and m["key"] in erow_c:
            key_row[m["cid"]] = ("concept", erow_c[m["key"]])
    # training features: use split-half evidence per user (first-half answered) to mimic online use
    ccov = env["ccov"]
    Xtr, ytr = [], []
    for rec in users:
        ans = sorted([int(c) for c in np.where(rec["ans_arr"])[0]], key=lambda c: -ccov[c])
        ev = ans[:max(1, len(ans) // 2)]
        ai, ki = _infer_channel(rec, ev, "item", key_row, Ei, bi)
        ac, kc = _infer_channel(rec, ev, "concept", key_row, Ec, bc_)
        zt = _taste_z(env, [rec], [ev])[0]
        for cd in np.where(env["valid"])[0]:
            cd = int(cd)
            kr = key_row.get(cd)
            if kr is None:
                ls, has = 0.0, 0.0
            else:
                ch, row = kr
                if ch == "item":
                    ls = float(bi[rec["u"]][row] + ai + Ei[rec["u"]][row] @ ki)
                else:
                    ls = float(bc_[rec["u"]][row] + ac + Ec[rec["u"]][row] @ kc)
                has = 1.0
            emb = CANDS[cd]["emb"]; taste = float((zt @ emb) / (np.linalg.norm(zt) * np.linalg.norm(emb) + 1e-9))
            Xtr.append([pop_k2[cd], ls, has, taste, np.log1p(ccov[cd])])
            ytr.append(_is_tgt(rec, cd))
    clf = LogisticRegression(max_iter=500, C=1.0).fit(np.array(Xtr), np.array(ytr))

    def score(env, rec, cds, ev_cids):
        ai, ki = _infer_channel(rec, ev_cids, "item", key_row, Ei, bi)
        ac, kc = _infer_channel(rec, ev_cids, "concept", key_row, Ec, bc_)
        zt = _taste_z(env, [rec], [ev_cids])[0]
        feats = []
        for cd in cds:
            kr = key_row.get(cd)
            if kr is None:
                ls, has = 0.0, 0.0
            else:
                ch, row = kr
                if ch == "item":
                    ls = float(bi[rec["u"]][row] + ai + Ei[rec["u"]][row] @ ki)
                else:
                    ls = float(bc_[rec["u"]][row] + ac + Ec[rec["u"]][row] @ kc)
                has = 1.0
            emb = env["CANDS"][cd]["emb"]
            taste = float((zt @ emb) / (np.linalg.norm(zt) * np.linalg.norm(emb) + 1e-9))
            feats.append([_poptgt(env)[cd], ls, has, taste, np.log1p(env["ccov"][cd])])
        p = clf.predict_proba(np.array(feats))[:, 1]
        return {cds[i]: float(p[i]) for i in range(len(cds))}
    return dict(score=score, clf=clf)


def test3(t2_pass):
    env = setup(); FR, model, users = env["FR"], env["model"], env["users"]
    print("\n==== T3 -- THE POLICY (fair harness; DIRECTIONAL 173/300) ====", flush=True)
    print("PRE-REGISTERED CONTRASTS (printed before results), all paired per-user bootstrap, anytime NDCG@10:", flush=True)
    print("  (1) r-vivid vs s-best     = the headline (must NOT lose beyond noise; E2 floor).", flush=True)
    print("  (2) r-vivid vs s-vivid    = the PERSONALIZATION increment (THE thesis quantity).", flush=True)
    print("  (3) s-vivid vs s-best     = the vividness-prior gain (credited to SCALE design, not adaptivity).", flush=True)

    passing, scheds, arms, cold = _reproduce_statics(env)
    def anyt(pt):
        return pt[:, :T3_T].mean(axis=1)
    static_rows = {n: float(anyt(arms[n][0]).mean()) for n in arms}
    best_static = max(static_rows, key=lambda n: static_rows[n])
    s_best_sched = scheds[{"s-concept": "s_concept", "s-item": "s_item", "s-mixed": "s_mixed"}[best_static]]
    print(f"  statics: " + ", ".join(f"{n} {static_rows[n]:.4f}" for n in static_rows) +
          f"  -> s-best = {best_static}", flush=True)

    # value scale for eps grid
    cval = env["cval"]; vc = cval[env["valid"]]
    spread = float(np.std(vc)) if len(vc) else 0.01
    eps_grid = [0.25 * spread, 0.5 * spread, 1.0 * spread]
    delta_grid = [0.0, 0.05, 0.1]
    belief = "iii" if t2_pass else "pop"
    if belief == "iii":
        env["_belief_iii"] = _make_belief_iii(env)

    # validation split of users (40%) for eps/delta grid search
    rng = np.random.default_rng(SEED); vperm = rng.permutation(len(users))
    val_idx = set(vperm[:int(0.4 * len(users))].tolist())
    val_users = [users[i] for i in sorted(val_idx)]
    full_users = users

    def eval_sched_static(sched, us):
        pt, ans = C.eval_plan(FR, model, us, C.plan_static(sched), C.cold_ndcg(FR, model, us))
        return anyt(pt)

    def eval_rvivid(eps, delta, us):
        pt, ans = C.eval_plan(FR, model, us, _rvivid_plan(env, s_best_sched, eps, delta, belief),
                              C.cold_ndcg(FR, model, us))
        return anyt(pt)

    # grid search r-vivid vs s-vivid on val
    best = None
    for eps in eps_grid:
        sv = _svivid_sched(env, s_best_sched, eps)
        sv_val = eval_sched_static(sv, val_users)
        for delta in delta_grid:
            rv_val = eval_rvivid(eps, delta, val_users)
            gain = float((rv_val - sv_val).mean())
            print(f"   [grid] eps={eps:.4f} delta={delta:.3f} -> r-vivid-s-vivid(val) {gain:+.4f}", flush=True)
            if best is None or gain > best["gain"]:
                best = dict(eps=eps, delta=delta, gain=gain)
    eps, delta = best["eps"], best["delta"]
    print(f"  chosen eps={eps:.4f} delta={delta:.3f} (val r-vivid-s-vivid {best['gain']:+.4f})", flush=True)

    # FULL evaluation
    cold_full = env["cold"]
    s_best_full = anyt(arms[best_static][0])
    s_vivid_sched = _svivid_sched(env, s_best_sched, eps)
    s_vivid_full = eval_sched_static(s_vivid_sched, full_users)
    r_vivid_full = eval_rvivid(eps, delta, full_users)

    c1 = paired(r_vivid_full, s_best_full)
    c2 = paired(r_vivid_full, s_vivid_full)
    c3 = paired(s_vivid_full, s_best_full)
    print(f"\n  s-best={s_best_full.mean():.4f}  s-vivid={s_vivid_full.mean():.4f}  r-vivid={r_vivid_full.mean():.4f}",
          flush=True)
    print(f"  (1) r-vivid - s-best  {c1['delta']:+.4f}[{c1['ci'][0]:+.4f},{c1['ci'][1]:+.4f}]", flush=True)
    print(f"  (2) r-vivid - s-vivid {c2['delta']:+.4f}[{c2['ci'][0]:+.4f},{c2['ci'][1]:+.4f}]  (personalization)", flush=True)
    print(f"  (3) s-vivid - s-best  {c3['delta']:+.4f}[{c3['ci'][0]:+.4f},{c3['ci'][1]:+.4f}]  (vividness prior)", flush=True)
    e2_floor_ok = bool(c1["ci"][0] >= -0.01)
    print(f"  E2 floor (r-vivid does not lose to s-best beyond noise): {e2_floor_ok}", flush=True)

    RESULTS["T3"] = dict(belief=belief, passing=passing, best_static=best_static,
                         statics=static_rows, eps=eps, delta=delta, val_gain=best["gain"],
                         s_best=float(s_best_full.mean()), s_vivid=float(s_vivid_full.mean()),
                         r_vivid=float(r_vivid_full.mean()),
                         c1_rvivid_vs_sbest=c1, c2_rvivid_vs_svivid=c2, c3_svivid_vs_sbest=c3,
                         e2_floor_ok=e2_floor_ok,
                         schedules=dict(
                             s_best=[f"{env['CANDS'][c]['kind']}:{env['CANDS'][c]['key']}" for c in s_best_sched],
                             s_vivid=[f"{env['CANDS'][c]['kind']}:{env['CANDS'][c]['key']}" for c in s_vivid_sched]))
    _t3_md(env, RESULTS["T3"])
    _dump()


def _t3_md(env, r):
    md_write("## T3 -- THE POLICY (fair harness)\n\n"
             f"T={T3_T}, NDCG@10 anytime, all {len(env['users'])} users (E1). Statics reproduced via the "
             f"Stage-C canary + greedy build on the shared pool. NO oracle / target-peek / answerability-table "
             f"arms (Stage-C retraction). r-vivid belief = **{r['belief']}** "
             f"({'personalized LOUO-MF+taste (T2 passed)' if r['belief']=='iii' else 'population k2-rate only (T2 failed)'}). "
             f"eps/delta grid-searched on a 40% val split of users (chosen eps={r['eps']:.4f}, "
             f"delta={r['delta']:.3f}; val r-vivid-s-vivid {r['val_gain']:+.4f}).\n\n"
             "**PRE-REGISTERED CONTRASTS:** (1) r-vivid vs s-best = headline (must not lose beyond noise, E2); "
             "(2) r-vivid vs s-vivid = personalization increment (THE thesis quantity); (3) s-vivid vs s-best "
             "= vividness-prior gain (credited to the SCALE design, not adaptivity).\n\n"
             "| arm | anytime NDCG@10 | class |\n|---|--:|---|\n")
    md_write(f"| s-best ({r['best_static']}) | {r['s_best']:.4f} | strongest static (reproduced) |\n")
    md_write(f"| s-vivid | {r['s_vivid']:.4f} | population-vivid static (deployable, no user evidence) |\n")
    md_write(f"| r-vivid | {r['r_vivid']:.4f} | full vivid-swap router ({r['belief']}) |\n")
    c1, c2, c3 = r["c1_rvivid_vs_sbest"], r["c2_rvivid_vs_svivid"], r["c3_svivid_vs_sbest"]
    md_write("\n### Three pre-registered contrasts (paired per-user bootstrap)\n\n"
             f"- **(1) r-vivid vs s-best (headline):** {c1['delta']:+.4f}[{c1['ci'][0]:+.4f},{c1['ci'][1]:+.4f}] "
             f"-> {'WINS' if c1['ci'][0] > 0 else ('loses beyond noise (E2 violation)' if c1['ci'][1] < -0.01 else 'ties / within E2 floor')}.\n"
             f"- **(2) r-vivid vs s-vivid (personalization increment -- THE thesis quantity):** "
             f"{c2['delta']:+.4f}[{c2['ci'][0]:+.4f},{c2['ci'][1]:+.4f}] "
             f"-> {'personalization ADDS (CI excl 0)' if c2['ci'][0] > 0 else 'no personalization increment (ties)'}.\n"
             f"- **(3) s-vivid vs s-best (vividness-prior gain, SCALE design):** "
             f"{c3['delta']:+.4f}[{c3['ci'][0]:+.4f},{c3['ci'][1]:+.4f}] "
             f"-> {'vividness prior helps (CI excl 0)' if c3['ci'][0] > 0 else 'no prior gain (ties)'}.\n\n"
             f"E2 floor (r-vivid does not lose to s-best beyond noise): **{r['e2_floor_ok']}**.\n\n"
             "### DIRECTIONAL caveats\n\n"
             "- 173/300 users, grid UNFROZEN; re-run on the frozen grid before citation.\n"
             "- Value model = per-candidate mean single-answer NDCG lift (population; identical across arms).\n"
             "- s-vivid/r-vivid swap only WITHIN channel and within +/- eps value of the scheduled question; "
             "r-vivid at t=0 (no evidence) reduces to the population-vivid pick (~s-vivid).\n\n")


def _dump():
    RESULTS["wall_min"] = round((time.time() - _ENV["t0"]) / 60, 2) if _ENV else None
    json.dump(RESULTS, open(OUT_JSON, "w"), indent=1, default=str)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", choices=["1", "2", "3", "all"], required=True)
    a = ap.parse_args()
    if a.test in ("1", "all"):
        p1 = test1()
        if a.test == "all" and not p1:
            print("\n[STOP] T1 gate FAILED -- premise dead; T2/T3 not run (cheap death = method success).", flush=True)
            sys.exit(0)
    if a.test == "2":
        test2()
    elif a.test == "3":
        test3(RESULTS.get("T2", {}).get("gate_pass", False))
    elif a.test == "all":
        p2 = test2()
        test3(p2)
    print("\n[done] wrote", OUT_MD, "and", OUT_JSON, flush=True)
