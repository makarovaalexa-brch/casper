"""tree_louo.py -- LEAVE-ONE-USER-OUT (K-fold) re-run of the LEARNED interview TREE vs LEARNED STATIC,
FOCUSED on the concept (s-mixed) family -- the live 6/6-positive-ns trend from TREE_VS_STATIC.md.

WHY: TREE_VS_STATIC grew each artifact on ONE HALF (86-87 construction users) and evaluated on the
OTHER. The concept trees TIED with a consistently positive point estimate (+0.0021 [-0.0007,+0.0049]
MU15 T12, 6/6 estimates positive, none significant) on BALANCED (66/20) answer-class splits. The
morning briefing's #1 live path: "concept-tree at power -- 6/6 positive, needs users." This script
DOUBLES the construction cohort by leave-one-user-out (172 construction users per eval user) or, if
that is computationally infeasible, the largest feasible K-fold (K-1 folds construct, ~156-172 users)
so the estimator is properly fed, and asks: does the trend become a DETECTABLE, adequately-powered
adaptivity advantage at n=173?

REUSE (verbatim): the grower + eval + identity proof come from learned_tree_vs_static.py (LT.grow_tree,
LT.tree_stats, LT.walk, LT.eval_tree_peruser, LT.greedy_pick, LT.answer_class, LT.tree_slim). The
static comparator is SC.build_greedy_sub / SC.eval_static_peruser (same greedy objective + code path;
branching-disabled tree == the sequence -- identity proof re-passes). The anchor gate re-passes.

PROTOCOL (K-fold cross-construction; every eval user scored by a tree/static grown on its complement):
- FOCUS family = s-mixed (concept-led). Item family = a small CHEAP confirmation at the end only.
- Partition the 173 users into K folds (seed 0). For fold k: construction cohort = the other K-1
  folds (156-172 users); GROW the concept tree AND the greedy concept static on that cohort; EVALUATE
  both on fold k. Same folds for tree and static (paired). Union of eval folds = all 173 users ->
  pooled paired per-user delta over n=173.
- MIN_USERS in {15, 30} (scales with the bigger cohort). Budgets T in {8, 12, 24}; trees grown to
  depth 24, statics prefix-consistent (one depth-24 build serves all budgets).
- LOUO (K=173) is measured first from one grow's wall time; if the extrapolation is infeasible the
  script drops to the largest K under the wall budget (>= 10) and SAYS SO in the report banner.

E-LEDGER (STATE_2026-07-08, E1-E7):
  E1 same-user paired bootstrap, same full 173-user set (every user is an eval user exactly once).
  E2 the static is the tree's floor by construction (branch-disabled tree == sequence); no-loss check.
  E3 no privileged arms (no oracle / target-peek / answerability table); both arms greedy-NDCG only.
  E5 cross-construction firewall: the tree/static that scores user u NEVER saw u (u is held out).
  E6 pre-registered thresholds + E7 arm-symmetry table printed BEFORE results; deterministic seed 0.
  E7 arm-symmetry: tree and static share cohort / objective / code path / fold / pool / eval; the ONLY
     difference is that the tree may branch on the observed answer class (Golbandi). Printed as a table.

NO LLM API calls; all local. DIRECTIONAL 173/300 (v2 fold .cache/i25_fold_v2_best.pt) -- re-run on the
frozen 300-user grid before any citation.

Run:  python scripts/tree_louo.py --do all            (K auto-selected under the wall budget)
      python scripts/tree_louo.py --k 10              (force 10-fold)
      python scripts/tree_louo.py --quick             (smoke: K=3, depth 12, MU15, concept only)
"""
import os, sys, json, time, argparse, collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings("ignore")
import repair_probes as RP
import static_contamination as SC
import learned_tree_vs_static as LT          # grower + eval reused VERBATIM

OUT_MD = "experiments/TREE_LOUO.md"
OUT_JSON = "experiments/tree_louo.json"
CKPT_JSON = "experiments/tree_louo.json.partial"
ANCHOR = 0.2251
ANCHOR_TOL = 0.002
TMAX = 24                       # depth = 24 turns (verdict budgets T=8/12/24)
BUDGETS = [8, 12, 24]
MIN_USERS_LIST = [15, 30]
SEED = 0
WALL_BUDGET_MIN = 240.0         # soft cap for the tree-grow campaign (K auto-selection target)
WALL_HARD_MIN = 420.0           # hard stop -> write whatever completed

RESULTS = {"banner": "DIRECTIONAL 173/300, grid unfrozen (v2 fold answerer-v1 working grid)"}


def md(txt, mode="a"):
    os.makedirs("experiments", exist_ok=True)
    open(OUT_MD, mode, encoding="utf-8").write(txt)


def ckpt():
    json.dump(RESULTS, open(CKPT_JSON, "w"), indent=1, default=str)


# ------------------------------------------------------------ folds
def make_folds(n, K, seed=SEED):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return [sorted(f.tolist()) for f in np.array_split(perm, K)]


def half_split(n, seed=SEED):
    """Reproduce TREE_VS_STATIC's seed-0 half split EXACTLY (default_rng(0).permutation, n//2)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    half = n // 2
    return sorted(perm[:half].tolist()), sorted(perm[half:].tolist())


# ------------------------------------------------------------ pos/neg follow-up readout (coarse->fine)
def followup_map(root, CANDS):
    """For each branching node, record the split question and the NEXT question asked in its pos child
    and in its neg child -- the coarse->fine visibility the author wants ('what gets asked after a
    positive vs a negative answer')."""
    rows = []

    def qof(node):
        return None if not node or node.get("question") is None else node.get("qlabel")

    def rec(node):
        if node.get("branch"):
            ch = node.get("children", {})
            rows.append(dict(depth=node["depth"], split=node["qlabel"],
                             n=node["n_construct"],
                             child_sizes=node.get("child_sizes"),
                             after_pos=qof(ch.get("pos")), after_neg=qof(ch.get("neg")),
                             after_noclue=qof(ch.get("noclue"))))
            for c in ch.values():
                rec(c)
        elif "child" in node:
            rec(node["child"])
    rec(root)
    return rows


def agg_struct(structs):
    """Aggregate tree-structure stats across folds: mean nodes/branched, branched-by-depth histogram,
    deepest branch, channel mix, mean fallbacks."""
    if not structs:
        return {}
    nodes = [s["n_nodes"] for s in structs]
    branched = [s["n_branched"] for s in structs]
    deepest = [s["max_depth_branch"] for s in structs]
    fbs = [s.get("_fallbacks", 0) for s in structs]
    bd = collections.Counter()
    for s in structs:
        for d, c in s["branched_by_depth"].items():
            bd[int(d)] += c
    ch = collections.Counter()
    for s in structs:
        for k, v in s["branch_channel_mix"].items():
            ch[k] += v
    return dict(n_folds=len(structs),
                mean_nodes=float(np.mean(nodes)), min_nodes=int(min(nodes)), max_nodes=int(max(nodes)),
                mean_branched=float(np.mean(branched)), min_branched=int(min(branched)),
                max_branched=int(max(branched)),
                mean_deepest_branch=float(np.mean(deepest)),
                total_branched_by_depth=dict(sorted(bd.items())),
                channel_mix=dict(ch), mean_fallbacks=float(np.mean(fbs)))


# ------------------------------------------------------------ pre-registered banner + E7 table
PREREG = (
    "## Pre-registered reads (printed BEFORE results)\n\n"
    "CLASS-MATCHED contrast at POWER: the LEARNED conditional interview TREE vs the LEARNED static\n"
    "questionnaire, IDENTICAL learner (same greedy NDCG objective + code path; branching-disabled tree\n"
    "== the sequence, asserted at runtime), grown by CROSS-CONSTRUCTION (K-fold: every eval user scored\n"
    "by an artifact grown on the OTHER K-1 folds = 156-172 construction users, ~2x the 86 of the\n"
    "half-split TREE_VS_STATIC run). FOCUS family = s-mixed (concept). Pooled paired per-user delta\n"
    "(tree - static) over ALL 173 users (each user an eval user exactly once), bootstrap CI, at\n"
    "T=8/12/24.\n\n"
    "THRESHOLDS (decided before any number is seen):\n"
    "- **CI EXCLUDES 0, POSITIVE** -> the program's FIRST FAIR, ADEQUATELY-POWERED ADAPTIVITY WIN:\n"
    "  the concept-tree's 6/6-positive-ns trend becomes detectable once the estimator is fed 2x data.\n"
    "  Stated plainly, no softening.\n"
    "- **CI INCLUDES 0** -> the trend does NOT strengthen with 2x construction data. Evidence the true\n"
    "  advantage is below the resolvable band; we QUANTIFY THE BOUND as the CI upper limit (report it).\n"
    "- **CI EXCLUDES 0, NEGATIVE** -> reported as-is (the tree loses even fed 2x data).\n\n"
    "SECONDARY (E7 -- both arms get the same extra food): the K-fold static (156-172 construction) vs\n"
    "the OLD half-split static (86 construction), paired per-user over 173 -> does the STATIC also\n"
    "improve with more data? Neither arm may be starved relative to the other.\n\n"
    "MECHANISM: branch counts / depths / channel mix at 156-172 users vs the old 86; which concept is\n"
    "asked after a positive vs a negative answer (coarse->fine visibility).\n\n"
    "GATES: (0) all-173 greedy s-item anytime@T=12 == 0.2251 +/- 0.002 (anchor). (1) branch-disabled\n"
    "tree schedule == SC.build_greedy_sub sequence (SAME learner). (E2) in-sample tree >= in-sample\n"
    "static by construction.\n\n"
)

E7_TABLE = (
    "## E7 ARM-SYMMETRY TABLE (what each arm sees / learns from / is constructed on)\n\n"
    "| property | LEARNED TREE | LEARNED STATIC (greedy sequence) | symmetric? |\n"
    "|---|---|---|---|\n"
    "| construction cohort | the eval user's complement (K-1 folds, 156-172 users) | SAME complement | YES |\n"
    "| objective | greedy cohort-mean NDCG@10 gain | greedy cohort-mean NDCG@10 gain | YES |\n"
    "| code path | LT.greedy_pick (per-node body) | SC.build_greedy_sub (per-pos body) -- identical | YES |\n"
    "| candidate pool | top-60 concepts by coverage | top-60 concepts by coverage | YES |\n"
    "| instrument / fold | v2 fold i25_fold_v2_best | v2 fold i25_fold_v2_best | YES |\n"
    "| eval cohort | held-out fold (never in construction) | SAME held-out fold | YES |\n"
    "| privileged info / value model | NONE | NONE | YES |\n"
    "| refusal handling | no-op turn, cold fallback (E1) | no-op turn, cold fallback (E1) | YES |\n"
    "| **conditioning on observed answer class** | **YES (3-way Golbandi branch, >=MIN_USERS)** | **NO (fixed order)** | **the ONLY asymmetry -- the treatment under test** |\n\n"
    "> The tree is a strict superset of the static (branch-disabled tree == the sequence). The single\n"
    "> asymmetry is the adaptivity itself. Any held-out tree advantage is therefore attributable to\n"
    "> answer-class conditioning and nothing else (E7 satisfied).\n\n"
)


# ------------------------------------------------------------ main
def run(quick=False, force_k=None):
    t0 = time.time()
    env = RP.setup()
    env = RP.value_tiers(env)
    users = env["users"]; n = len(users)
    CANDS = env["CANDS"]

    cold_all = RP.cold_ndcg(env["FR"], users)
    for i, rec in enumerate(users):
        rec["_cold"] = float(cold_all[i])

    item_pool = [m["cid"] for m in CANDS if m["kind"] == "item"]
    concept_pool = [m["cid"] for m in CANDS if m["kind"] == "concept"]
    mixed_pool = [m["cid"] for m in CANDS]

    global TMAX
    if quick:
        TMAX = 12

    # header
    md("# LEARNED TREE vs LEARNED STATIC -- LEAVE-ONE-USER-OUT / K-FOLD (concept family; DIRECTIONAL 173/300)\n\n", mode="w")
    md("> **DIRECTIONAL BANNER** -- 173/300 users, answerer-v1 working grid NOT frozen; v2 fold\n"
       "> `.cache/i25_fold_v2_best.pt`. NO LLM API calls; local compute. Re-run on the frozen 300-user\n"
       "> grid before any citation.\n\n"
       f"Date 2026-07-09. Script `scripts/tree_louo.py`. {n} users; {len(CANDS)} candidates\n"
       f"({sum(m['kind']=='concept' for m in CANDS)}c/{sum(m['kind']=='item' for m in CANDS)}i/"
       f"{sum(m['kind']=='attr' for m in CANDS)}a); cold NDCG@10 {cold_all.mean():.4f}; fold val "
       f"{env['foldval']:.4f}. Depth={TMAX} turns. Candidate pool = top-{RP.POOL} by coverage (same as\n"
       f"the static). Paired per-user bootstrap BOOT={RP.BOOT}. FOCUS = s-mixed (concept) family; item\n"
       f"family = cheap confirmation only. Answer classes: pos(liked/loved)/neg(meh/hated)/"
       f"noclue(refuse,k=0; E1).\n\n")
    md(PREREG)
    md(E7_TABLE)
    print("\n" + PREREG, flush=True)
    print(E7_TABLE, flush=True)
    RESULTS["prereg"] = "printed"
    RESULTS["tmax"] = TMAX

    # ---- 0. anchor gate (SC.build_greedy_sub, all users) ----
    print("==== 0. ANCHOR GATE (all-%d greedy s-item anytime@T12) ====" % n, flush=True)
    repro_sched = SC.build_greedy_sub(env, item_pool, users, cold_all, TMAX)
    repro_pt = SC.eval_static_peruser(env, repro_sched, users, cold_all, TMAX)
    repro_any12 = float(repro_pt[:, :12].mean())
    okrepro = abs(repro_any12 - ANCHOR) <= ANCHOR_TOL
    print(f"  anytime@T12 = {repro_any12:.4f} (anchor {ANCHOR} +/- {ANCHOR_TOL}) -> "
          f"{'MATCH' if okrepro else 'MISMATCH'}", flush=True)
    md("## 0. Reproduction (anchor gate)\n\n"
       f"All-{n} greedy s-item anytime NDCG@10 @T=12 = **{repro_any12:.4f}** vs anchor {ANCHOR} +/- "
       f"{ANCHOR_TOL} -> **{'MATCH' if okrepro else 'MISMATCH'}**.\n\n")
    RESULTS["reproduce"] = dict(anytime_T12=repro_any12, anchor=ANCHOR, match=bool(okrepro))
    ckpt()
    if not okrepro:
        md("**STOP: reproduction failed; LOUO not run.**\n\n")
        json.dump(RESULTS, open(OUT_JSON, "w"), indent=1, default=str)
        print("[STOP] reproduction mismatch.", flush=True); return

    # ---- 1. identity proof (branch-disabled tree == greedy sequence) ----
    print("\n==== 1. IDENTITY PROOF (branch-disabled tree == static sequence) ====", flush=True)
    nid = [0]
    id_root = LT.grow_tree(env, sorted(item_pool, key=lambda c: -CANDS[c]["cov"])[:RP.POOL],
                           users, [], 0, TMAX, LT.NO_BRANCH, nid)
    seq = []; cur = id_root
    while cur is not None and not cur.get("leaf") and cur.get("question") is not None:
        seq.append(cur["question"]); cur = cur.get("child")
    identity_ok = (seq == list(repro_sched[:TMAX]))
    print(f"  no-branch tree schedule == SC.build_greedy_sub schedule ? {identity_ok}", flush=True)
    md("## 1. Identity proof (SAME learner)\n\n"
       f"Branching-disabled tree schedule == greedy sequence (`SC.build_greedy_sub`) ? **{identity_ok}**.\n"
       "Certifies the tree grower and the static comparator share one code path / objective.\n\n")
    RESULTS["identity_ok"] = bool(identity_ok)
    ckpt()
    if not identity_ok:
        md("**STOP: learner mismatch. Fix before results.**\n\n")
        json.dump(RESULTS, open(OUT_JSON, "w"), indent=1, default=str)
        print("[STOP] identity mismatch.", flush=True); return

    # ---- 2. TIMING PROBE -> choose K ----
    print("\n==== 2. TIMING PROBE (one concept grow) -> choose protocol ====", flush=True)
    K0 = 3 if quick else 10
    probe_folds = make_folds(n, K0)
    probe_con = [i for i in range(n) if i not in set(probe_folds[0])]
    probe_users = [users[i] for i in probe_con]
    t_probe = time.time()
    nid = [0]
    LT.GROW_CKPT.update(root=None, count=0, path=None)
    _ = LT.grow_tree(env, sorted(concept_pool, key=lambda c: -CANDS[c]["cov"])[:RP.POOL],
                     probe_users, [], 0, TMAX, MIN_USERS_LIST[0], nid)
    t_grow = time.time() - t_probe
    print(f"  one concept grow on {len(probe_users)} users, depth {TMAX} = {t_grow:.1f}s", flush=True)

    n_min = 1 if quick else len(MIN_USERS_LIST)
    # LOUO extrapolation (grows scale ~linearly in cohort size; 172/156 ~= 1.10x)
    louo_grows = n * n_min
    louo_min = louo_grows * t_grow * (172.0 / max(len(probe_users), 1)) / 60.0
    print(f"  LOUO (K={n}) extrapolation: {louo_grows} grows ~= {louo_min:.0f} min "
          f"({louo_min/60:.1f} h)", flush=True)

    if force_k is not None:
        K = force_k
        why_k = f"forced via --k {force_k}"
    elif quick:
        K = 3; why_k = "quick smoke"
    else:
        # largest K in {15,12,10} whose concept-grow campaign fits the wall budget; floor 10.
        K = 10; why_k = ""
        for cand in (15, 12, 10):
            grows = cand * n_min
            est = grows * t_grow * (172.0 / max(len(probe_users), 1)) / 60.0
            if est <= WALL_BUDGET_MIN:
                K = cand
                why_k = (f"largest K with concept campaign ~{est:.0f} min <= {WALL_BUDGET_MIN:.0f} min "
                         f"budget; LOUO (K={n}) would be ~{louo_min:.0f} min -> INFEASIBLE")
                break
        else:
            why_k = (f"10-fold floor (even K=10 est ~{10*n_min*t_grow*(172.0/max(len(probe_users),1))/60.0:.0f} "
                     f"min); LOUO ~{louo_min:.0f} min INFEASIBLE")
    con_size = n - int(np.ceil(n / K))
    print(f"  -> PROTOCOL: {K}-fold ({why_k}); construction cohort ~= {con_size} users "
          f"(vs 86 half-split, vs 172 LOUO)", flush=True)
    md("## 2. Protocol selection (LOUO vs K-fold)\n\n"
       f"One concept grow (depth {TMAX}, {len(probe_users)} users) = **{t_grow:.1f}s**. Full LOUO "
       f"(K={n}, {louo_grows} grows) extrapolates to **~{louo_min:.0f} min ({louo_min/60:.1f} h)**.\n\n"
       f"**PROTOCOL USED: {K}-fold cross-construction** ({why_k}). Construction cohort ~= "
       f"**{con_size} users** per eval user (vs 86 in the half-split TREE_VS_STATIC run; vs 172 for "
       f"full LOUO). Every one of the {n} users is an eval user in exactly one fold -> pooled n={n}.\n\n")
    RESULTS["protocol"] = dict(K=K, why=why_k, t_grow_s=t_grow, louo_est_min=louo_min,
                               construction_users=con_size, n_min=n_min)
    ckpt()

    folds = make_folds(n, K)
    fold_of = {}
    for k, f in enumerate(folds):
        for i in f:
            fold_of[i] = k

    # ---- 3. OLD half-split static per user (E7 baseline: 86-user construction) ----
    print("\n==== 3. OLD half-split static per user (86-user construction; E7 baseline) ====", flush=True)
    A_idx, B_idx = half_split(n)
    A_users = [users[i] for i in A_idx]; B_users = [users[i] for i in B_idx]
    coldA = np.array([users[i]["_cold"] for i in A_idx])
    coldB = np.array([users[i]["_cold"] for i in B_idx])
    half_static = {}   # fam -> {T -> np.array(n)}
    for fam, pool in (("s-mixed", concept_pool), ("s-item", item_pool)):
        sA = SC.build_greedy_sub(env, pool, A_users, coldA, TMAX)
        sB = SC.build_greedy_sub(env, pool, B_users, coldB, TMAX)
        ptB = SC.eval_static_peruser(env, sA, B_users, coldB, TMAX)   # B scored by A-grown (86)
        ptA = SC.eval_static_peruser(env, sB, A_users, coldA, TMAX)   # A scored by B-grown (86)
        arr = {T: np.full(n, np.nan) for T in BUDGETS}
        for j, i in enumerate(B_idx):
            for T in BUDGETS:
                arr[T][i] = ptB[j, :T].mean()
        for j, i in enumerate(A_idx):
            for T in BUDGETS:
                arr[T][i] = ptA[j, :T].mean()
        half_static[fam] = arr
        print(f"  [{fam}] half-split static any@12 mean = {np.nanmean(arr[12]):.4f}", flush=True)

    # ---- 4. K-FOLD GROW + EVAL ----
    # storage: per family -> tree[mu][T][i], static[T][i]
    def new_store():
        return {T: np.full(n, np.nan) for T in BUDGETS}
    families_to_run = [("s-mixed", concept_pool)]   # focus; item confirmation appended later if cheap
    RESULTS["folds"] = {}
    struct_by_fam_mu = collections.defaultdict(list)     # (fam,mu) -> [struct]
    followups = collections.defaultdict(list)            # (fam,mu) -> list of followup rows (few folds)
    tree_store = collections.defaultdict(dict)           # (fam) -> {mu -> store}
    static_store = {}                                    # fam -> store

    def run_family(fam, pool, mus, tag_cheap=False):
        pool_sorted = sorted(pool, key=lambda c: -CANDS[c]["cov"])[:RP.POOL]
        sstore = new_store()
        tstores = {mu: new_store() for mu in mus}
        # -- pass A: build + eval the K statics ONCE (cheap; guarantees the E7 baseline exists) --
        fold_ctx = []
        for k in range(K):
            ev_idx = folds[k]
            con_idx = [i for i in range(n) if fold_of[i] != k]
            con_users = [users[i] for i in con_idx]
            ev_users = [users[i] for i in ev_idx]
            con_cold = np.array([users[i]["_cold"] for i in con_idx])
            ev_cold = np.array([users[i]["_cold"] for i in ev_idx])
            s_sched = SC.build_greedy_sub(env, pool, con_users, con_cold, TMAX)
            spt = SC.eval_static_peruser(env, s_sched, ev_users, ev_cold, TMAX)
            spt_in = SC.eval_static_peruser(env, s_sched, con_users, con_cold, TMAX)
            for j, i in enumerate(ev_idx):
                for T in BUDGETS:
                    sstore[T][i] = spt[j, :T].mean()
            fold_ctx.append(dict(k=k, ev_idx=ev_idx, con_users=con_users, ev_users=ev_users,
                                 in_static12=float(spt_in[:, :12].mean())))
        static_store[fam] = sstore
        print(f"  [{fam}] K={K} statics built; K-fold static any@12 = {np.nanmean(sstore[12]):.4f}",
              flush=True)
        ckpt()
        # -- pass B: grow trees, MU-OUTER so the headline MU (mus[0]) finishes for ALL folds first --
        for mu in mus:
            for fc in fold_ctx:
                if (time.time() - t0) / 60.0 > WALL_HARD_MIN:
                    print(f"  [HARD WALL {WALL_HARD_MIN}m] stopping {fam} MU{mu} at fold {fc['k']}",
                          flush=True)
                    break
                k = fc["k"]; ev_idx = fc["ev_idx"]
                con_users = fc["con_users"]; ev_users = fc["ev_users"]
                nid = [0]
                LT.GROW_CKPT.update(root=None, count=0,
                                    path=f".cache/tree_louo_grow_{fam}_MU{mu}_k{k}.json")
                root = LT.grow_tree(env, pool_sorted, con_users, [], 0, TMAX, mu, nid)
                LT.GROW_CKPT.update(root=None, path=None)
                st = LT.tree_stats(root)
                tpt, _, tfb = LT.eval_tree_peruser(env, root, ev_users, TMAX)
                tpt_in, _, _ = LT.eval_tree_peruser(env, root, con_users, TMAX)
                st["_fallbacks"] = int(tfb)
                struct_by_fam_mu[(fam, mu)].append(st)
                if k < 3:  # a few folds' worth of coarse->fine readout
                    followups[(fam, mu)].append(dict(fold=k, con=len(con_users),
                                                     rows=followup_map(root, CANDS)[:14]))
                for j, i in enumerate(ev_idx):
                    for T in BUDGETS:
                        tstores[mu][T][i] = tpt[j, :T].mean()
                in_tree = float(tpt_in[:, :12].mean()); in_stat = fc["in_static12"]
                san = in_tree >= in_stat - 1e-9
                json.dump({"fold": k, "fam": fam, "mu": mu, "tree": LT.tree_slim(root)},
                          open(f".cache/tree_louo_ckpt_{fam}_MU{mu}_k{k}.json", "w"),
                          indent=1, default=str)
                RESULTS["folds"][f"{fam}_MU{mu}_k{k}"] = dict(
                    con=len(con_users), ev=len(ev_idx), branched=st["n_branched"],
                    nodes=st["n_nodes"], fallbacks=int(tfb),
                    in_tree=in_tree, in_static=in_stat, e2_sanity=bool(san),
                    ho_tree12=float(np.nanmean([tstores[mu][12][i] for i in ev_idx])),
                    ho_static12=float(np.nanmean([sstore[12][i] for i in ev_idx])))
                print(f"  [{fam} MU{mu} fold {k}] con={len(con_users)} ev={len(ev_idx)} "
                      f"branched={st['n_branched']} nodes={st['n_nodes']} fb={tfb} "
                      f"in(tree>=stat)={san} | HO@12 tree "
                      f"{np.nanmean([tstores[mu][12][i] for i in ev_idx]):.4f} vs static "
                      f"{np.nanmean([sstore[12][i] for i in ev_idx]):.4f}", flush=True)
                # tree_store must reflect completed folds even on early stop
                tree_store[fam] = tstores
                ckpt()
        tree_store[fam] = tstores

    run_family("s-mixed", concept_pool, MIN_USERS_LIST if not quick else [15])

    # cheap item confirmation only if we have plenty of wall budget left
    elapsed = (time.time() - t0) / 60.0
    do_item = (not quick) and (elapsed + K * t_grow * (172.0 / max(len(probe_users), 1)) / 60.0
                               < WALL_HARD_MIN)
    if do_item:
        print("\n==== item-family confirmation (MU15 only; the settled family) ====", flush=True)
        run_family("s-item", item_pool, [15])
    else:
        print(f"\n[skip item confirmation] elapsed {elapsed:.0f}m; not cheap enough.", flush=True)

    # ---- 5. VERDICTS ----
    RESULTS["verdicts"] = {}
    RESULTS["static_improvement"] = {}
    RESULTS["struct_agg"] = {}
    RESULTS["struct_old_86"] = {  # from TREE_VS_STATIC.md section 4 (concept trees, 86-user halves)
        "s-mixed": "nodes 63-109; branched 11-16; deepest branch depth 10; channel mix concept-only"}
    for fam in tree_store:
        for mu in tree_store[fam]:
            RESULTS["struct_agg"][f"{fam}_MU{mu}"] = agg_struct(struct_by_fam_mu[(fam, mu)])
        # tree - kfold static (headline), each budget
        for mu in tree_store[fam]:
            for T in BUDGETS:
                tv = list(tree_store[fam][mu][T]); sv = list(static_store[fam][T])
                cb = RP.paired(tv, sv)
                RESULTS["verdicts"][f"{fam}_MU{mu}_T{T}"] = cb
        # static improvement: kfold static vs old half-split static (E7)
        for T in BUDGETS:
            cb = RP.paired(list(static_store[fam][T]), list(half_static[fam][T]))
            RESULTS["static_improvement"][f"{fam}_T{T}"] = cb

    # primary verdict = s-mixed MU15 T12
    prim = RESULTS["verdicts"].get("s-mixed_MU15_T12")
    lo, hi = prim["ci"]
    if lo > 0:
        vtext = ("FIRST FAIR ADEQUATELY-POWERED ADAPTIVITY WIN -- the concept-tree trend becomes "
                 "detectable at 2x construction data")
    elif hi < 0:
        vtext = "TREE LOSES even fed 2x data (CI excl 0, negative) -- reported as-is"
    else:
        vtext = (f"TIE -- the trend does NOT strengthen with 2x data; true advantage bounded above by "
                 f"the CI upper limit {hi:+.4f} (< ~0.005 if within band)")
    RESULTS["verdict"] = dict(primary="s-mixed_MU15_T12", pooled=prim, text=vtext)

    write_md(env, tree_store, static_store, half_static, struct_by_fam_mu, followups)
    RESULTS["wall_min"] = round((time.time() - t0) / 60, 2)
    json.dump(RESULTS, open(OUT_JSON, "w"), indent=1, default=str)

    print("\n==== VERDICT ====", flush=True)
    print(f"  PRIMARY (s-mixed MU15 T12, pooled n={prim['n']}): tree - static = "
          f"{prim['delta']:+.4f}[{lo:+.4f},{hi:+.4f}]", flush=True)
    print(f"  -> {vtext}", flush=True)
    print(f"[done] wrote {OUT_MD} + {OUT_JSON} (wall {RESULTS['wall_min']}m)", flush=True)
    return RESULTS


def write_md(env, tree_store, static_store, half_static, struct_by_fam_mu, followups):
    md("## 3. HEADLINE -- tree vs K-fold static, pooled paired per-user (n=173)\n\n"
       "Each eval user scored by a tree AND a static grown on its complement (same fold, paired).\n\n"
       "| family | MIN_USERS | budget | tree - static [95% CI] | n | CI excl 0? | verdict |\n"
       "|---|--:|--:|---|--:|---|---|\n")
    for key in sorted(RESULTS["verdicts"].keys()):
        cb = RESULTS["verdicts"][key]
        fam, mu, T = key.rsplit("_", 2)
        excl = (cb["ci"][0] > 0 or cb["ci"][1] < 0)
        vd = ("WIN" if (excl and cb["delta"] > 0) else ("LOSS" if excl else "tie"))
        md(f"| {fam} | {mu[2:]} | {T[1:]} | {cb['delta']:+.4f}[{cb['ci'][0]:+.4f},{cb['ci'][1]:+.4f}] | "
           f"{cb['n']} | {'YES' if excl else 'no'} | {vd} |\n")
    v = RESULTS["verdict"]
    md(f"\n**PRIMARY VERDICT (s-mixed MU15 T12): tree - static = "
       f"{v['pooled']['delta']:+.4f}[{v['pooled']['ci'][0]:+.4f},{v['pooled']['ci'][1]:+.4f}] "
       f"(n={v['pooled']['n']}) -> {v['text']}.**\n\n"
       "Comparison to the half-split run (TREE_VS_STATIC.md): concept MU15 T12 pooled was "
       "**+0.0021 [-0.0007,+0.0049]** at 86 construction users (6/6 estimates positive, none "
       "significant). This row is the same contrast fed ~2x the construction data.\n\n")

    md("## 4. STATIC IMPROVEMENT (E7 -- both arms get the same extra food)\n\n"
       "K-fold static (156-172 construction users) vs the OLD half-split static (86 construction), "
       "paired per-user over all 173. Positive = the STATIC also improves with more construction data "
       "(the extra food is not tree-only).\n\n"
       "| family | budget | K-fold static - half-split static [95% CI] | n | K-fold mean | half mean |\n"
       "|---|--:|---|--:|--:|--:|\n")
    for fam in static_store:
        for T in BUDGETS:
            cb = RESULTS["static_improvement"][f"{fam}_T{T}"]
            km = float(np.nanmean(static_store[fam][T])); hm = float(np.nanmean(half_static[fam][T]))
            md(f"| {fam} | {T} | {cb['delta']:+.4f}[{cb['ci'][0]:+.4f},{cb['ci'][1]:+.4f}] | {cb['n']} | "
               f"{km:.4f} | {hm:.4f} |\n")
    md("\n")

    md("## 5. TREE STRUCTURE at power (156-172 users) vs the old 86\n\n"
       "Aggregated across the K folds. OLD (half-split, 86 users, TREE_VS_STATIC.md sec 4, concept): "
       "nodes 63-109; branched 11-16; deepest branch depth 10; concept-only.\n\n"
       "| family | MIN_USERS | folds | mean nodes | mean branched | max branched | deepest branch | mean fallbacks |\n"
       "|---|--:|--:|--:|--:|--:|--:|--:|\n")
    for key in sorted(RESULTS["struct_agg"].keys()):
        s = RESULTS["struct_agg"][key]
        if not s:
            continue
        fam, mu = key.rsplit("_MU", 1)
        md(f"| {fam} | {mu} | {s['n_folds']} | {s['mean_nodes']:.0f} | {s['mean_branched']:.1f} | "
           f"{s['max_branched']} | {s['mean_deepest_branch']:.1f} | {s['mean_fallbacks']:.1f} |\n")
    md("\n### Branching-by-depth histogram (summed across folds)\n\n")
    for key in sorted(RESULTS["struct_agg"].keys()):
        s = RESULTS["struct_agg"][key]
        if not s:
            continue
        md(f"- **{key}**: channel mix {s['channel_mix']}; branched-by-depth "
           f"{s['total_branched_by_depth']}\n")
    md("\n")

    md("## 6. COARSE->FINE READOUT (what is asked after a positive vs a negative answer)\n\n"
       "First folds' concept trees; for each branch: the split concept, cohort sizes (pos/neg/noclue), "
       "and the NEXT concept asked down the positive vs the negative child.\n\n")
    for key in sorted(followups.keys()):
        fam, mu = key
        for blk in followups[key][:1]:   # one representative fold per (fam,mu)
            md(f"### {fam} MU{mu} (fold {blk['fold']}, {blk['con']} construction users)\n\n"
               "| depth | split concept | pos/neg/noclue | after POS | after NEG |\n"
               "|--:|---|---|---|---|\n")
            for r in blk["rows"]:
                cs = r.get("child_sizes") or {}
                csx = "/".join(str(cs.get(x, 0)) for x in ("pos", "neg", "noclue"))
                md(f"| {r['depth']} | {r['split']} | {csx} | {r['after_pos'] or '-'} | "
                   f"{r['after_neg'] or '-'} |\n")
            md("\n")

    md("## Synthesis\n\n")
    v = RESULTS["verdict"]
    si = RESULTS["static_improvement"].get("s-mixed_T12", {})
    md(f"LOUO/K-fold, concept family, class-matched, cross-constructed, held-out. Primary pooled verdict "
       f"(s-mixed MU15 T12, n={v['pooled']['n']}): **{v['pooled']['delta']:+.4f}"
       f"[{v['pooled']['ci'][0]:+.4f},{v['pooled']['ci'][1]:+.4f}]** -> **{v['text']}**. "
       f"Static-improvement (K-fold vs half-split, T12): "
       f"{si.get('delta', float('nan')):+.4f}[{si.get('ci',[float('nan')]*2)[0]:+.4f},"
       f"{si.get('ci',[float('nan')]*2)[1]:+.4f}] (E7: both arms fed the same extra data). "
       f"DIRECTIONAL 173/300; re-run on the frozen grid before citation.\n\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--do", choices=["all"], default="all")
    ap.add_argument("--k", type=int, default=None, help="force K (else auto-selected under wall budget)")
    ap.add_argument("--quick", action="store_true", help="smoke: K=3, depth 12, MU15, concept only")
    a = ap.parse_args()
    run(quick=a.quick, force_k=a.k)
