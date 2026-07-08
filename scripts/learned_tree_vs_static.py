"""learned_tree_vs_static.py -- CLASS-MATCHED adaptivity: a LEARNED conditional interview TREE vs the
LEARNED static questionnaire, IDENTICAL learner, split-constructed, evaluated on HELD-OUT users.

WHY: every prior contrast pitted a LEARNED static (greedy direct search with outcome labels on the
cohort) against HAND-DESIGNED adaptive rules -- an unfair class mismatch. This gives both sides the
SAME learner. The static comparator is the greedy 12-question SEQUENCE grown on the same construction
half with the SAME code path; the tree is the Golbandi-style ternary generalization of that greedy
that, with branching DISABLED, reproduces the sequence EXACTLY (identity proof asserted at runtime).

DESIGN (Golbandi-style ternary tree, learned exactly like the static):
- SPLIT users into halves A/B (seed 0; seed 1 if time). GROW on A, EVAL both artifacts on B; swap.
- TREE: at each node, greedy-select the question maximizing construction-cohort mean NDCG@10 gain
  (SAME objective/code path as the static -- SC.build_greedy_sub's per-position body), then branch the
  cohort on the observed ANSWER CLASS: {pos (liked/loved), neg (meh/hated), noclue (refuse/k=0)}.
  Split a node ONLY if >= MIN_USERS construction users reach it (else it continues as an unbranched
  sequence -- prevents overfit leaves). Depth = 12 turns. Refusal (k=0) = no fold update, turn
  consumed, user descends the noclue branch (E1).
- EVAL on the held-out half: every user WALKS the tree per their own answers; anytime NDCG@10 + endpoint
  at T=8/12; paired per-user bootstrap vs the grow-half static evaluated on the same held-out users.
  ALSO vs the eval-half-grown static (in-sample reference; contextualizes contamination).

HONESTY: both artifacts learn ONLY from construction-half users; no privileged info. Candidate pool =
top-POOL(60) by coverage (same as the static); documented. NO LLM API calls; all local. DIRECTIONAL
173/300 (v2 fold .cache/i25_fold_v2_best.pt) -- re-run on the frozen 300-user grid before citation.

Answer-class rule (documented): noclue iff not answered (k=0); else pos iff centered fold value > 0
(loved/liked), neg iff <= 0 (meh/hated). Centered fold values live in rec["val_arr"] (concept/attr map
via CENTERED_FOLD {hated -1, meh -1/3, liked +1/3, loved +1}; items = stars - user-mean).

Run:  python scripts/learned_tree_vs_static.py --do all
"""
import os, sys, json, time, argparse, collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings("ignore")
import repair_probes as RP
import static_contamination as SC

OUT_MD = "experiments/TREE_VS_STATIC.md"
OUT_JSON = "experiments/tree_vs_static.json"
ANCHOR = 0.2251          # all-173 greedy s-item anytime@T=12 (FOLD_V2_REPAIR / STATIC_CONTAMINATION)
ANCHOR_TOL = 0.002
TMAX = 12                 # depth = 12 turns
BUDGETS = [8, 12]
MIN_USERS_LIST = [15, 25]
SEEDS = [0, 1]
NO_BRANCH = 10 ** 9       # MIN_USERS sentinel that disables branching (identity proof)

RESULTS = {"banner": "DIRECTIONAL 173/300, grid unfrozen (v2 fold answerer-v1 working grid)"}


def md(txt, mode="a"):
    os.makedirs("experiments", exist_ok=True)
    open(OUT_MD, mode, encoding="utf-8").write(txt)


# ------------------------------------------------------------ answer class
def answer_class(rec, cid):
    if not rec["ans_arr"][cid]:
        return "noclue"
    return "pos" if float(rec["val_arr"][cid]) > 0 else "neg"


# ------------------------------------------------------------ node greedy pick (SC.build_greedy_sub body)
def greedy_pick(env, pool, sub_users, sub_cold, asked):
    """Greedy-select the single candidate maximizing cohort-mean NDCG@10 over sub_users given the
    already-asked path (fold = answered path questions + candidate if answered; cold fallback for users
    who answer nothing). IDENTICAL logic to SC.build_greedy_sub's per-position body -> the branch-off
    sequence reproduces the static exactly."""
    FR, model, CANDS = env["FR"], env["model"], env["CANDS"]
    n = len(sub_users)
    cands = [c for c in pool if c not in asked]
    if not cands:
        return None
    tl, nl, idx, owner = [], [], [], []
    base_ans = [[c for c in asked if rec["ans_arr"][c]] for rec in sub_users]
    for ci, c in enumerate(cands):
        for i, rec in enumerate(sub_users):
            cids = base_ans[i] + ([c] if rec["ans_arr"][c] else [])
            if not cids:
                continue
            tl.append([RP.tok_of(CANDS, rec, cc) for cc in cids])
            nl.append([rec["nat_arr"][cc] for cc in cids if rec["nat_arr"][cc] is not None])
            idx.append(i); owner.append(ci)
    ndcg = RP.batch_ndcg(FR, model, sub_users, tl, nl, idx)
    sums = np.zeros(len(cands)); got = collections.defaultdict(set)
    for r, ci in enumerate(owner):
        sums[ci] += ndcg[r]; got[ci].add(idx[r])
    best, bc = -1.0, None
    for ci, c in enumerate(cands):
        mean = (sums[ci] + sum(sub_cold[i] for i in range(n) if i not in got[ci])) / n
        if mean > best:
            best, bc = mean, c
    return bc


# ------------------------------------------------------------ tree grower
GROW_CKPT = {"root": None, "path": None, "every": 8, "count": 0}


def _grow_ckpt():
    """Persist the partial tree during growth (interruption loses <= GROW_CKPT['every'] node picks)."""
    GROW_CKPT["count"] += 1
    if GROW_CKPT["path"] and GROW_CKPT["root"] is not None \
            and GROW_CKPT["count"] % GROW_CKPT["every"] == 0:
        try:
            json.dump(tree_slim(GROW_CKPT["root"]), open(GROW_CKPT["path"], "w"),
                      indent=1, default=str)
        except Exception:
            pass


def grow_tree(env, pool, sub_users, asked, depth, T, min_users, node_id):
    """Recursively grow the interview tree. Returns a node dict.
    Branch (3-way on answer class) iff len(cohort) >= min_users; else continue as unbranched sequence.
    min_users = NO_BRANCH sentinel disables branching -> the returned schedule IS the greedy sequence."""
    node = {"id": node_id[0], "depth": depth, "n_construct": len(sub_users)}
    node_id[0] += 1
    if depth >= T or not sub_users:
        node["question"] = None; node["leaf"] = True; node["branch"] = False
        return node
    sub_cold = [rec["_cold"] for rec in sub_users]
    cid = greedy_pick(env, pool, sub_users, sub_cold, asked)
    node["question"] = cid
    if GROW_CKPT["root"] is None:
        GROW_CKPT["root"] = node  # first (root) node of this grow
    _grow_ckpt()
    node["qlabel"] = None if cid is None else f"{env['CANDS'][cid]['kind']}:{env['CANDS'][cid]['key']}"
    node["leaf"] = False
    if cid is None:
        node["branch"] = False; node["child"] = grow_tree(env, pool, sub_users, asked, depth + 1, T, min_users, node_id)
        return node
    asked2 = asked + [cid]
    if len(sub_users) >= min_users and depth < T - 1:
        # branch 3-way on the observed answer class
        groups = collections.defaultdict(list)
        for rec in sub_users:
            groups[answer_class(rec, cid)].append(rec)
        node["branch"] = True
        node["child_sizes"] = {cls: len(g) for cls, g in groups.items()}
        node["children"] = {}
        for cls, g in groups.items():  # attach incrementally so the grow checkpoint sees progress
            node["children"][cls] = grow_tree(env, pool, g, asked2, depth + 1, T, min_users, node_id)
    else:
        node["branch"] = False
        node["child"] = grow_tree(env, pool, sub_users, asked2, depth + 1, T, min_users, node_id)
    return node


def tree_slim(node):
    """JSON-serializable copy of a tree (drop nothing but keep it plain)."""
    out = {k: node[k] for k in ("id", "depth", "n_construct", "question", "qlabel", "leaf", "branch")
           if k in node}
    if node.get("branch"):
        out["child_sizes"] = node.get("child_sizes")
        out["children"] = {cls: tree_slim(c) for cls, c in node["children"].items()}
    elif "child" in node:
        out["child"] = tree_slim(node["child"])
    return out


def tree_stats(root):
    """Collect branching structure: branched-node count, by depth, and the split questions."""
    stats = {"n_nodes": 0, "n_branched": 0, "branched_by_depth": collections.Counter(),
             "branch_questions": [], "max_depth_branch": -1}
    ch_ch = {}

    def rec(node):
        stats["n_nodes"] += 1
        if node.get("branch"):
            stats["n_branched"] += 1
            stats["branched_by_depth"][node["depth"]] += 1
            stats["max_depth_branch"] = max(stats["max_depth_branch"], node["depth"])
            k = node["qlabel"].split(":")[0]
            ch_ch[k] = ch_ch.get(k, 0) + 1
            stats["branch_questions"].append(dict(depth=node["depth"], q=node["qlabel"],
                                                  n=node["n_construct"], child_sizes=node.get("child_sizes")))
            for c in node["children"].values():
                rec(c)
        elif "child" in node:
            rec(node["child"])
    rec(root)
    stats["branched_by_depth"] = dict(stats["branched_by_depth"])
    stats["branch_channel_mix"] = ch_ch
    return stats


# ------------------------------------------------------------ walk a held-out user through the tree
def walk(root, rec):
    """Return the ordered list of asked cids (the path this user takes). Missing-class branch (a class
    absent in construction) falls back to the LARGEST available child (logged via _fallbacks)."""
    path = []; cur = root; fb = 0
    while cur is not None and not cur.get("leaf") and cur.get("question") is not None:
        cid = cur["question"]; path.append(cid)
        if cur.get("branch"):
            cls = answer_class(rec, cid)
            nxt = cur["children"].get(cls)
            if nxt is None:
                fb += 1
                # fallback: descend the most-populated construction child
                sib = cur.get("children") or {}
                if not sib:
                    break
                cls_fb = max(sib, key=lambda k: cur["child_sizes"].get(k, 0))
                nxt = sib[cls_fb]
            cur = nxt
        else:
            cur = cur.get("child")
    return path, fb


def eval_tree_peruser(env, root, sub_users, T):
    FR, model, CANDS = env["FR"], env["model"], env["CANDS"]
    n = len(sub_users)
    cold = np.array([rec["_cold"] for rec in sub_users])
    paths = []; fbs = 0
    for rec in sub_users:
        p, fb = walk(root, rec); fbs += fb
        paths.append(p)
    per_turn = np.empty((n, T))
    for t in range(T):
        col = cold.copy(); tl, nl, idx = [], [], []
        for i, rec in enumerate(sub_users):
            cids = [c for c in paths[i][:t + 1] if (c is not None and rec["ans_arr"][c])]
            if not cids:
                continue
            tl.append([RP.tok_of(CANDS, rec, c) for c in cids])
            nl.append([rec["nat_arr"][c] for c in cids if rec["nat_arr"][c] is not None])
            idx.append(i)
        if tl:
            col[idx] = RP.batch_ndcg(FR, model, sub_users, tl, nl, idx)
        per_turn[:, t] = col
    return per_turn, paths, fbs


# ------------------------------------------------------------ pre-registered banner
PREREG = (
    "## Pre-registered reads (printed BEFORE results)\n\n"
    "CLASS-MATCHED contrast: a LEARNED conditional interview TREE vs the LEARNED static questionnaire,\n"
    "IDENTICAL learner (same greedy objective + code path; branching-disabled tree == the sequence,\n"
    "asserted at runtime), split-constructed (grown on one half), evaluated on the OTHER (held-out) half.\n\n"
    "- **(1) VERDICT -- tree vs (grow-half) static on HELD-OUT users** (anytime NDCG@10, pooled over both\n"
    "  split directions, paired per-user bootstrap). CI EXCLUDES 0 is THE verdict:\n"
    "    * POSITIVE -> first honest LEARNED-ADAPTIVITY win (conditioning transfers to new users).\n"
    "    * NEGATIVE/TIE -> statics optimal even class-matched (a real property, proven fairly).\n"
    "- **(2) IN-SAMPLE SANITY**: in-sample tree >= in-sample static (grow-half, both) -- MUST hold by\n"
    "  construction (the tree is a superset of the sequence). If violated, the learner differs -> STOP.\n"
    "- **(3) TREE STRUCTURE**: #nodes that branched, at which depths, on which questions/channels --\n"
    "  the interpretability payload (does it descend into taste regions after positive answers?).\n\n"
    "Also reported: tree & grow-half static vs the EVAL-HALF-grown static (in-sample reference) -- puts\n"
    "the (contaminated) in-sample static beside the fair one so the contamination gap is visible.\n"
    "MIN_USERS in {15, 25}; primary family s-item (matches anchor 0.2251); s-mixed secondary. Verdict\n"
    "metric = anytime NDCG@10 @T=12. Reproduction gate: all-173 greedy s-item anytime@T=12 == 0.2251.\n\n"
)


# ------------------------------------------------------------ main
def run(quick=False):
    t0 = time.time()
    env = RP.setup()
    env = RP.value_tiers(env)
    users = env["users"]; n = len(users)
    CANDS = env["CANDS"]

    # per-user cold (independent of cohort) attached once -> aligned sub_cold everywhere + static identity
    cold_all = RP.cold_ndcg(env["FR"], users)
    for i, rec in enumerate(users):
        rec["_cold"] = float(cold_all[i])

    item_pool = [m["cid"] for m in CANDS if m["kind"] == "item"]
    mixed_pool = [m["cid"] for m in CANDS]
    families = {"s-item": item_pool, "s-mixed": mixed_pool}
    seeds = [0] if quick else SEEDS

    md("# LEARNED INTERVIEW TREE vs LEARNED STATIC -- CLASS-MATCHED ADAPTIVITY (v2 fold; DIRECTIONAL 173/300)\n\n", mode="w")
    md("> **DIRECTIONAL BANNER** -- 173/300 users, answerer-v1 working grid NOT frozen; v2 fold\n"
       "> `.cache/i25_fold_v2_best.pt`. NO LLM API calls; local compute. Re-run on the frozen 300-user\n"
       "> grid before any citation.\n\n"
       f"Date 2026-07-09. Script `scripts/learned_tree_vs_static.py`. {n} users; {len(CANDS)} candidates\n"
       f"({sum(m['kind']=='concept' for m in CANDS)}c/{sum(m['kind']=='item' for m in CANDS)}i/"
       f"{sum(m['kind']=='attr' for m in CANDS)}a); cold NDCG@10 {cold_all.mean():.4f}; fold val "
       f"{env['foldval']:.4f}. Depth={TMAX} turns. Candidate pool = top-{RP.POOL} by coverage (same as the\n"
       f"static). Paired per-user bootstrap BOOT={RP.BOOT}. Answer classes: pos(liked/loved)/neg(meh/\n"
       f"hated)/noclue(refuse,k=0; turn consumed, no fold update, E1).\n\n")
    md(PREREG)
    print("\n" + PREREG, flush=True)

    # ---- 0. REPRODUCE (anchor gate) via SC.build_greedy_sub ----
    print("==== 0. REPRODUCE (all-%d greedy s-item, eval all) ====" % n, flush=True)
    repro_sched = SC.build_greedy_sub(env, item_pool, users, cold_all, TMAX)
    repro_pt = SC.eval_static_peruser(env, repro_sched, users, cold_all, TMAX)
    repro_any12 = float(repro_pt[:, :12].mean())
    okrepro = abs(repro_any12 - ANCHOR) <= ANCHOR_TOL
    print(f"  all-{n} s-item anytime@T12 = {repro_any12:.4f} (anchor {ANCHOR} +/- {ANCHOR_TOL}) -> "
          f"{'MATCH' if okrepro else 'MISMATCH'}", flush=True)
    md("## 0. Reproduction (anchor gate)\n\n"
       f"All-{n} greedy s-item anytime NDCG@10 @T=12 = **{repro_any12:.4f}** vs anchor {ANCHOR} +/- "
       f"{ANCHOR_TOL} -> **{'MATCH' if okrepro else 'MISMATCH'}**.\n\n")
    RESULTS["reproduce"] = dict(anytime_T12=repro_any12, anchor=ANCHOR, match=bool(okrepro))
    if not okrepro:
        md("**STOP: reproduction failed; tree-vs-static not run.**\n\n")
        json.dump(RESULTS, open(OUT_JSON, "w"), indent=1, default=str)
        print("[STOP] reproduction mismatch.", flush=True)
        return

    # ---- 1. IDENTITY PROOF: branching-disabled tree == greedy sequence ----
    print("\n==== 1. IDENTITY PROOF (branch-disabled tree == static sequence) ====", flush=True)
    nid = [0]
    id_root = grow_tree(env, sorted(item_pool, key=lambda c: -CANDS[c]["cov"])[:RP.POOL],
                        users, [], 0, TMAX, NO_BRANCH, nid)
    # extract the linear schedule from the no-branch tree
    seq = []; cur = id_root
    while cur is not None and not cur.get("leaf") and cur.get("question") is not None:
        seq.append(cur["question"]); cur = cur.get("child")
    identity_ok = (seq == list(repro_sched[:TMAX]))
    print(f"  no-branch tree schedule == SC.build_greedy_sub schedule ? {identity_ok}", flush=True)
    print(f"    tree: {[CANDS[c]['kind']+':'+str(CANDS[c]['key']) for c in seq]}", flush=True)
    md("## 1. Identity proof (SAME learner)\n\n"
       f"Branching-disabled tree schedule == greedy sequence (`SC.build_greedy_sub`) ? **{identity_ok}**.\n"
       "This certifies the tree grower and the static comparator share one code path / objective; the\n"
       "tree is the strict Golbandi-style branching superset of the sequence.\n\n"
       f"- schedule: `{[CANDS[c]['kind']+':'+str(CANDS[c]['key']) for c in seq]}`\n\n")
    RESULTS["identity_ok"] = bool(identity_ok)
    if not identity_ok:
        md("**STOP: learner mismatch (branch-disabled tree != sequence). Fix before results.**\n\n")
        json.dump(RESULTS, open(OUT_JSON, "w"), indent=1, default=str)
        print("[STOP] identity mismatch.", flush=True)
        return

    # ---- 2. GROW + EVAL over seeds x families x MIN_USERS ----
    RESULTS["runs"] = {}
    pooled = {}  # (fam, mu, T) -> dict(tree=[], stat=[])  (held-out, pooled over both directions)
    struct_dump = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n); half = n // 2
        A_idx = sorted(perm[:half].tolist()); B_idx = sorted(perm[half:].tolist())
        A_users = [users[i] for i in A_idx]; B_users = [users[i] for i in B_idx]
        coldA = np.array([r["_cold"] for r in A_users]); coldB = np.array([r["_cold"] for r in B_users])

        # split-constructed statics (grown on each half) via the shared SC machinery
        statics = {}
        for fam, pool in families.items():
            statics[(fam, "A")] = SC.build_greedy_sub(env, pool, A_users, coldA, TMAX)
            statics[(fam, "B")] = SC.build_greedy_sub(env, pool, B_users, coldB, TMAX)

        for fam, pool in families.items():
            pool_sorted = sorted(pool, key=lambda c: -CANDS[c]["cov"])[:RP.POOL]
            mus = MIN_USERS_LIST if (fam == "s-item" or seed == 0) else [15]
            if quick:
                mus = [15]
            for mu in mus:
                for grow, ev in (("A", "B"), ("B", "A")):
                    g_users = A_users if grow == "A" else B_users
                    e_users = B_users if ev == "B" else A_users
                    print(f"  [seed{seed} {fam} MU{mu}] grow {grow} (n={len(g_users)}) -> eval {ev} "
                          f"(n={len(e_users)}) ...", flush=True)
                    nid = [0]
                    GROW_CKPT.update(root=None, count=0,
                                     path=f".cache/tree_grow_partial_seed{seed}_{fam}_MU{mu}_{grow}.json")
                    root = grow_tree(env, pool_sorted, g_users, [], 0, TMAX, mu, nid)
                    GROW_CKPT.update(root=None, path=None)
                    st = tree_stats(root)
                    # tree eval on held-out
                    tpt, tpaths, tfb = eval_tree_peruser(env, root, e_users, TMAX)
                    # tree eval in-sample (sanity)
                    tpt_in, _, _ = eval_tree_peruser(env, root, g_users, TMAX)
                    # static grown on grow-half (fair) eval on held-out + in-sample
                    s_fair = statics[(fam, grow)]
                    spt = SC.eval_static_peruser(env, s_fair, e_users,
                                                 (coldB if ev == "B" else coldA), TMAX)
                    spt_in = SC.eval_static_peruser(env, s_fair, g_users,
                                                    (coldA if grow == "A" else coldB), TMAX)
                    # static grown on the EVAL half (in-sample reference / contaminated)
                    s_ref = statics[(fam, ev)]
                    spt_ref = SC.eval_static_peruser(env, s_ref, e_users,
                                                     (coldB if ev == "B" else coldA), TMAX)

                    rk = f"seed{seed}_{fam}_MU{mu}_grow{grow}_eval{ev}"
                    row = {"held_out": {}, "in_sample": {}, "vs_ref": {}, "fallbacks": int(tfb),
                           "structure": {k: v for k, v in st.items() if k != "branch_questions"},
                           "branch_questions": st["branch_questions"]}
                    for T in BUDGETS:
                        tv = list(tpt[:, :T].mean(axis=1)); sv = list(spt[:, :T].mean(axis=1))
                        tiv = list(tpt_in[:, :T].mean(axis=1)); siv = list(spt_in[:, :T].mean(axis=1))
                        rv = list(spt_ref[:, :T].mean(axis=1))
                        row["held_out"][str(T)] = dict(tree=float(np.mean(tv)), static=float(np.mean(sv)),
                                                       delta=RP.paired(tv, sv))
                        row["in_sample"][str(T)] = dict(tree=float(np.mean(tiv)), static=float(np.mean(siv)),
                                                        delta=RP.paired(tiv, siv))
                        row["vs_ref"][str(T)] = dict(tree_vs_ref=RP.paired(tv, rv),
                                                     fairstat_vs_ref=RP.paired(sv, rv),
                                                     ref_mean=float(np.mean(rv)))
                        pooled.setdefault((fam, mu, T), {"tree": [], "stat": []})
                        pooled[(fam, mu, T)]["tree"].extend(tv)
                        pooled[(fam, mu, T)]["stat"].extend(sv)
                    RESULTS["runs"][rk] = row
                    struct_dump[rk] = st
                    # INCREMENTAL CHECKPOINT after EVERY split-estimate: an interruption loses at
                    # most one estimate (coordinator requirement, overnight robustness).
                    json.dump(RESULTS, open(OUT_JSON + ".partial", "w"), indent=1, default=str)
                    json.dump({"run": rk, "tree": tree_slim(root)},
                              open(f".cache/tree_ckpt_{rk}.json", "w"), indent=1, default=str)
                    md(f"<!-- partial: {rk} held-out@12 tree "
                       f"{row['held_out']['12']['tree']:.4f} vs static "
                       f"{row['held_out']['12']['static']:.4f} -->\n")
                    d12 = row["held_out"]["12"]["delta"]
                    ins = row["in_sample"]["12"]
                    print(f"    HELD-OUT@12 tree {row['held_out']['12']['tree']:.4f} vs static "
                          f"{row['held_out']['12']['static']:.4f}  delta {d12['delta']:+.4f}"
                          f"[{d12['ci'][0]:+.4f},{d12['ci'][1]:+.4f}]", flush=True)
                    print(f"    IN-SAMPLE@12 tree {ins['tree']:.4f} vs static {ins['static']:.4f} "
                          f"(tree>=static? {ins['tree'] >= ins['static'] - 1e-9})", flush=True)
                    print(f"    branched nodes {st['n_branched']} (by depth {st['branched_by_depth']}); "
                          f"channel mix {st['branch_channel_mix']}; fallbacks {tfb}", flush=True)

    # ---- pooled verdicts ----
    RESULTS["pooled"] = {}
    for (fam, mu, T), d in pooled.items():
        RESULTS["pooled"][f"{fam}_MU{mu}_T{T}"] = RP.paired(d["tree"], d["stat"])

    # primary verdict: s-item, MU=15, T=12, pooled over both directions (seed 0 required; seed1 adds n)
    prim = RESULTS["pooled"].get("s-item_MU15_T12")
    ci_excl = bool(prim["ci"][0] > 0 or prim["ci"][1] < 0)
    if not ci_excl:
        verdict = "TIE -> statics optimal even class-matched (fair proof)"
    elif prim["delta"] > 0:
        verdict = "TREE WINS -> first honest learned-adaptivity win (conditioning transfers)"
    else:
        verdict = "TREE LOSES -> statics strictly optimal even class-matched"
    RESULTS["verdict"] = dict(primary="s-item_MU15_T12", pooled=prim, ci_excludes_0=ci_excl, text=verdict)

    write_md(env, families, seeds)
    RESULTS["wall_min"] = round((time.time() - t0) / 60, 2)
    json.dump(RESULTS, open(OUT_JSON, "w"), indent=1, default=str)

    print("\n==== VERDICT ====", flush=True)
    print(f"  PRIMARY (s-item MU15 T12, pooled both directions): tree-static = "
          f"{prim['delta']:+.4f}[{prim['ci'][0]:+.4f},{prim['ci'][1]:+.4f}] (n={prim['n']})", flush=True)
    print(f"  -> {verdict}", flush=True)
    print(f"\n[done] wrote {OUT_MD} + {OUT_JSON} (wall {RESULTS['wall_min']}m)", flush=True)
    return RESULTS


def write_md(env, families, seeds):
    R = RESULTS["runs"]
    md("## 2. HELD-OUT verdict (tree vs grow-half static; class-matched, split-constructed)\n\n")
    # pooled table first (the verdict)
    md("### Pooled over both split directions (THE verdict)\n\n"
       "| family | MIN_USERS | budget | tree - static [95% CI] | n | CI excl 0? |\n"
       "|---|--:|--:|---|--:|---|\n")
    for key in sorted(RESULTS["pooled"].keys()):
        cb = RESULTS["pooled"][key]
        fam, mu, T = key.rsplit("_", 2)
        excl = (cb["ci"][0] > 0 or cb["ci"][1] < 0)
        md(f"| {fam} | {mu[2:]} | {T[1:]} | {cb['delta']:+.4f}[{cb['ci'][0]:+.4f},{cb['ci'][1]:+.4f}] | "
           f"{cb['n']} | {'YES' if excl else 'no'} |\n")
    v = RESULTS["verdict"]
    md(f"\n**PRIMARY VERDICT (s-item MU15 T12): tree - static = "
       f"{v['pooled']['delta']:+.4f}[{v['pooled']['ci'][0]:+.4f},{v['pooled']['ci'][1]:+.4f}] "
       f"-> {v['text']}**\n\n")

    md("### Per split direction (held-out)\n\n"
       "| run | budget | tree | static(fair) | tree-static [CI] | in-sample tree | in-sample static | "
       "sanity tree>=static | fallbacks |\n|---|--:|--:|--:|---|--:|--:|:--:|--:|\n")
    for rk in sorted(R.keys()):
        row = R[rk]
        for T in BUDGETS:
            h = row["held_out"][str(T)]; ins = row["in_sample"][str(T)]
            san = "OK" if ins["tree"] >= ins["static"] - 1e-9 else "**VIOLATED**"
            md(f"| {rk} | {T} | {h['tree']:.4f} | {h['static']:.4f} | "
               f"{h['delta']['delta']:+.4f}[{h['delta']['ci'][0]:+.4f},{h['delta']['ci'][1]:+.4f}] | "
               f"{ins['tree']:.4f} | {ins['static']:.4f} | {san} | {row['fallbacks']} |\n")
    md("\n")

    md("## 3. Contamination context -- both vs the EVAL-HALF-grown static (in-sample reference)\n\n"
       "The eval-half-grown static is fit to the very users it is scored on (contaminated); comparing the\n"
       "fair (grow-half) static to it exposes the test-fit bonus, and the tree to it shows whether\n"
       "adaptivity closes that gap.\n\n"
       "| run | budget | ref(eval-grown) static | tree - ref [CI] | fair-static - ref [CI] |\n"
       "|---|--:|--:|---|---|\n")
    for rk in sorted(R.keys()):
        row = R[rk]
        for T in BUDGETS:
            vr = row["vs_ref"][str(T)]
            tvr = vr["tree_vs_ref"]; fvr = vr["fairstat_vs_ref"]
            md(f"| {rk} | {T} | {vr['ref_mean']:.4f} | "
               f"{tvr['delta']:+.4f}[{tvr['ci'][0]:+.4f},{tvr['ci'][1]:+.4f}] | "
               f"{fvr['delta']:+.4f}[{fvr['ci'][0]:+.4f},{fvr['ci'][1]:+.4f}] |\n")
    md("\n")

    md("## 4. TREE STRUCTURE (interpretability payload)\n\n")
    for rk in sorted(R.keys()):
        row = R[rk]; st = row["structure"]
        if "MU15" not in rk and "MU25" not in rk:
            continue
        md(f"### {rk}\n\n"
           f"- nodes {st['n_nodes']}; branched {st['n_branched']} "
           f"(by depth {st['branched_by_depth']}); deepest branch depth {st['max_depth_branch']}; "
           f"branch channel mix {st['branch_channel_mix']}; fallbacks {row['fallbacks']}\n\n")
        bq = row["branch_questions"]
        if bq:
            md("| depth | split question | cohort n | child sizes (pos/neg/noclue) |\n|--:|---|--:|---|\n")
            for b in bq[:24]:
                cs = b.get("child_sizes") or {}
                csx = "/".join(str(cs.get(k, 0)) for k in ("pos", "neg", "noclue"))
                md(f"| {b['depth']} | {b['q']} | {b['n']} | {csx} |\n")
            if len(bq) > 24:
                md(f"| ... | ({len(bq)-24} more branch nodes) | | |\n")
            md("\n")

    md("## Synthesis\n\n")
    v = RESULTS["verdict"]
    md(f"Class-matched, split-constructed, held-out: the LEARNED interview tree vs the LEARNED static\n"
       f"(identity-proven same learner). Primary pooled verdict (s-item MU15 T12): "
       f"**{v['pooled']['delta']:+.4f}[{v['pooled']['ci'][0]:+.4f},{v['pooled']['ci'][1]:+.4f}]** "
       f"-> **{v['text']}**. DIRECTIONAL 173/300; re-run on the frozen grid before citation.\n\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--do", choices=["all"], default="all")
    ap.add_argument("--quick", action="store_true", help="seed 0 only, MU15 only (smoke test)")
    ap.parse_args()
    run(quick=ap.parse_args().quick)
