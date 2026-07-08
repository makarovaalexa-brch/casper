"""ANSWERER v1 -- FULL 300-USER RUN (parallelized). Author-directed 2026-07-08.

Runs the VALIDATED small-batch machinery (scripts/answerer_smallbatch.py) over ALL 300 study users
(the answerability-gate cohort = .cache/instrument2/answerability_grid_ml25m.json users, intersect the
pinned seed-123 answerer split). Judging is BYTE-IDENTICAL to the small batch: this module imports
answerer_smallbatch as SB and calls SB.profile_lines / SB.build_questions / SB.run_production /
SB.run_validation / SB.parse_compact / SB.SYS_PROD / SB.SYS_VALID. Only the DRIVER changes:

  * ALL 300 users (the 10 small-batch users are ABSORBED at merge from answerer_v1_grid10.json --
    never re-judged).
  * PARALLEL: --worker K --nshards N judges users where (index in sorted cohort) % N == K; each worker
    owns answerer_v1_shard{K}.json + answerer_v1_shard{K}_usage.json; incremental flush per user;
    independently restartable (resumes from its own shard, cumulative usd carried in the sidecar).
  * --merge : shards + grid10 -> answerer_v1_grid300.json + answerer_v1_valid300.json (QUARANTINED) +
    answerer_v1_usage300.json. Detects + reports any user judged twice (keeps first, logs it).
  * --costcheck : sums the usage sidecars, prints projected total (monitor loop uses this; kill if >$20).
  * --splitsens : Hole-5 closure; 30 users x seeds {124,125} x reduced battery (200 tags + 100 items).
  * --report : experiments/ANSWERER_V1_FINAL.md + freeze manifest (sha256).

Judge gpt-5.4-mini-2026-03-17 temp 0, split seed 123. Deterministic seeds throughout. ASCII output.
"""
import os, sys, json, argparse, collections, hashlib, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings("ignore")
import answerer_smallbatch as SB          # VALIDATED judging machinery (imports G = pinned judge)
G = SB.G

CACHE = ".cache/instrument2"
GATE_GRID = f"{CACHE}/answerability_grid_ml25m.json"
GRID10 = f"{CACHE}/answerer_v1_grid10.json"
VALID10 = f"{CACHE}/answerer_v1_valid.json"
USAGE10 = f"{CACHE}/answerer_v1_grid10_usage.json"

GRID300 = f"{CACHE}/answerer_v1_grid300.json"
VALID300 = f"{CACHE}/answerer_v1_valid300.json"
USAGE300 = f"{CACHE}/answerer_v1_usage300.json"
SPLITSENS = f"{CACHE}/answerer_v1_splitsens.json"
SPLITSENS_USAGE = f"{CACHE}/answerer_v1_splitsens_usage.json"
REPORT = "experiments/ANSWERER_V1_FINAL.md"

SPLIT_SEED = G.ANSWERER_SPLIT_SEED         # 123
ALT_SEEDS = [124, 125]
N_SS_USERS = 30
N_SS_TAGS = 200
N_SS_ITEMS = 100
BUDGET_CAP = 20.0

KNOW = SB.KNOW
VAL = SB.VAL


def shard_path(k):
    return f"{CACHE}/answerer_v1_shard{k}.json"


def shard_usage_path(k):
    return f"{CACHE}/answerer_v1_shard{k}_usage.json"


# ------------------------------------------------------------------ cohort (300 study users)
def load_cohort(split):
    gate = json.load(open(GATE_GRID))["users"]
    return sorted(int(u) for u in gate if int(u) in split)


def grid10_users():
    return set(int(u) for u in json.load(open(GRID10))["users"])


def size_terciles(D, cohort):
    sizes = [len(dict(D["rat_by_u"][u])) for u in cohort]
    return np.quantile(sizes, [1 / 3, 2 / 3])


# ------------------------------------------------------------------ one user (mirror of SB.collect body)
def judge_user(u, split, D, battery, tags, top800, qs, tokstat):
    """Judge ONE user with the EXACT small-batch pipeline. Returns (grid_entry, valid_entry)."""
    kn, ho = split[u]
    rat = dict(D["rat_by_u"][u])
    N_total = len(rat)
    plines, capped = SB.profile_lines(D, kn, rat)
    n_shown = len(plines)
    prof = "; ".join(plines) if plines else "(no ratings shown)"
    if capped:
        prof = "[a star-stratified sample of this viewer's ratings]\n" + prof
    Q, data_cells, held_in_top800, heldset = SB.build_questions(u, split, D, battery, tags, top800)

    ans = SB.run_production(u, split, D, Q, prof, n_shown, N_total, tokstat)
    vrows = SB.run_validation(u, split, D, prof, n_shown, N_total, tokstat)

    cell_out = {}
    item_ids_in_grid = set()
    for i, (k, m, _txt) in enumerate(Q):
        o = ans.get(i)
        rec = dict(channel=k, ans=o if o is not None else None)
        if k == "concept":
            rec["tagId"] = int(m["tagId"]); rec["tag"] = m["tag"]
        elif k == "attribute":
            rec["entity_id"] = m["entity_id"]; rec["atype"] = m["atype"]
        elif k == "item":
            rec["j"] = int(m["j"]); item_ids_in_grid.add(int(m["j"]))
        cell_out[str(i)] = rec
    data_out = {}
    for j, c in data_cells.items():
        data_out[str(j)] = dict(channel="item", j=int(j), **c)
        item_ids_in_grid.add(int(j))

    # *** LEAKAGE ASSERTION (identical to small batch) ***
    leak = item_ids_in_grid & heldset
    assert not leak, f"LEAKAGE: held-out item ids entered grid for user {u}: {sorted(leak)[:10]}"

    dom, _ = G.dominant_genre(D, sorted(kn), rat)
    sz = 0 if N_total <= qs[0] else (1 if N_total <= qs[1] else 2)
    grid_entry = dict(cell=[int(sz), str(dom)], dom=str(dom), n_shown=n_shown, N_total=N_total,
                      profile_capped=bool(capped), n_data_cells=len(data_out), n_llm_cells=len(cell_out),
                      n_heldout=len(heldset), heldout_in_top800_excluded=len(held_in_top800),
                      Q=cell_out, data=data_out)
    valid_entry = dict(n=len(vrows), rows=vrows)
    return grid_entry, valid_entry


# ------------------------------------------------------------------ worker
def run_worker(k, nshards):
    D = G.load_data()
    split = G.build_split(D)
    battery, tags, top800 = SB.load_inputs()
    cohort = load_cohort(split)
    qs = size_terciles(D, cohort)
    skip = grid10_users()
    assigned = [u for i, u in enumerate(cohort) if i % nshards == k]
    todo = [u for u in assigned if u not in skip]
    print(f"[worker {k}/{nshards}] assigned {len(assigned)} users, {len(assigned)-len(todo)} are grid10 "
          f"(absorbed at merge), {len(todo)} to judge", flush=True)

    sp = shard_path(k); up = shard_usage_path(k)
    grid = {}
    valid = {}
    base = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "usd": 0.0}
    tokstat = {"completion": 0, "prompt": 0, "questions": 0, "calls": 0}
    if os.path.exists(sp):
        b = json.load(open(sp))
        if b.get("split_seed") == SPLIT_SEED and b.get("model") == G.MODEL:
            grid = b.get("users", {}); valid = b.get("valid", {})
            tokstat = b.get("tokstat", tokstat)
    if os.path.exists(up):
        b = json.load(open(up))
        base = {kk: b.get(kk, 0) for kk in ("calls", "prompt_tokens", "completion_tokens", "usd")}
    print(f"[worker {k}] resuming {len(grid)} cached users (prior ${base['usd']:.4f})", flush=True)

    def flush():
        json.dump(dict(split_seed=SPLIT_SEED, model=G.MODEL, snapshot=G._USAGE["snapshot"], shard=k,
                       nshards=nshards, n_tags=len(tags), n_attrs=len(battery), n_top800=len(top800),
                       tokstat=tokstat, users=grid, valid=valid), open(sp, "w"))
        cum = {kk: base[kk] + G._USAGE[kk] for kk in ("calls", "prompt_tokens", "completion_tokens", "usd")}
        cum["snapshot"] = G._USAGE["snapshot"]; cum["shard"] = k
        cum["users_done"] = len(grid); cum["tokstat"] = tokstat
        json.dump(cum, open(up, "w"), indent=1)

    for u in todo:
        if str(u) in grid:
            continue
        ge, ve = judge_user(u, split, D, battery, tags, top800, qs, tokstat)
        grid[str(u)] = ge; valid[str(u)] = ve
        flush()
        cum_usd = base["usd"] + G._USAGE["usd"]
        print(f"  [w{k} {len(grid)}/{len(todo)}] u{u} dom={ge['dom']} LLMq={ge['n_llm_cells']} "
              f"data={ge['n_data_cells']} valid={ve['n']} capped={ge['profile_capped']} "
              f"| shard-cum ${cum_usd:.4f}", flush=True)
    flush()
    print(f"[worker {k}] DONE {len(grid)}/{len(todo)} judged; shard-cum "
          f"${base['usd'] + G._USAGE['usd']:.4f}", flush=True)


# ------------------------------------------------------------------ cost check (monitor loop)
def costcheck():
    total = 0.0; parts = {}
    for up in sorted(glob.glob(f"{CACHE}/answerer_v1_shard*_usage.json")):
        b = json.load(open(up)); total += b.get("usd", 0.0)
        parts[os.path.basename(up)] = dict(usd=round(b.get("usd", 0.0), 4),
                                            users_done=b.get("users_done"))
    # grid10 already-spent (absorbed) + splitsens if present
    g10 = json.load(open(USAGE10)).get("usd", 0.0) if os.path.exists(USAGE10) else 0.0
    ss = json.load(open(SPLITSENS_USAGE)).get("usd", 0.0) if os.path.exists(SPLITSENS_USAGE) else 0.0
    grand = total + g10 + ss
    out = dict(shards_usd=round(total, 4), grid10_usd=round(g10, 4), splitsens_usd=round(ss, 4),
               grand_total_usd=round(grand, 4), budget_cap=BUDGET_CAP,
               over_budget=bool(grand > BUDGET_CAP), shards=parts)
    print(json.dumps(out))
    return out


# ------------------------------------------------------------------ merge
def merge():
    split_meta = dict(split_seed=SPLIT_SEED, model=G.MODEL)
    grid = {}; valid = {}
    dup = []
    origin = {}
    snapshot = None
    tokstat = {"completion": 0, "prompt": 0, "questions": 0, "calls": 0}
    usd = 0.0; calls = 0; ptok = 0; ctok = 0

    # 1) absorb the 10 small-batch users FIRST (they win any dup, per "keep first")
    g10 = json.load(open(GRID10))
    snapshot = g10.get("snapshot")
    for u, rec in g10["users"].items():
        grid[u] = rec; origin[u] = "grid10"
    v10 = json.load(open(VALID10))["users"]
    for u, rec in v10.items():
        valid[u] = rec
    u10 = json.load(open(USAGE10))
    usd += u10.get("usd", 0.0); calls += u10.get("calls", 0)
    ptok += u10.get("prompt_tokens", 0); ctok += u10.get("completion_tokens", 0)
    for kk in tokstat:
        tokstat[kk] += u10.get("tokstat", {}).get(kk, 0)

    # 2) merge shards
    for sp in sorted(glob.glob(f"{CACHE}/answerer_v1_shard*.json")):
        if sp.endswith("_usage.json"):
            continue
        b = json.load(open(sp))
        if snapshot is None:
            snapshot = b.get("snapshot")
        for u, rec in b.get("users", {}).items():
            if u in grid:
                dup.append((u, origin.get(u, "?"), os.path.basename(sp)))
                continue                     # keep first
            grid[u] = rec; origin[u] = os.path.basename(sp)
        for u, rec in b.get("valid", {}).items():
            if u not in valid:
                valid[u] = rec
        ts = b.get("tokstat", {})
        for kk in tokstat:
            tokstat[kk] += ts.get(kk, 0)
    for up in sorted(glob.glob(f"{CACHE}/answerer_v1_shard*_usage.json")):
        b = json.load(open(up))
        usd += b.get("usd", 0.0); calls += b.get("calls", 0)
        ptok += b.get("prompt_tokens", 0); ctok += b.get("completion_tokens", 0)

    json.dump(dict(**split_meta, snapshot=snapshot, n_users=len(grid), tokstat=tokstat,
                   users=grid), open(GRID300, "w"))
    json.dump(dict(QUARANTINE="held-out-half predictions -- NEVER feed to the environment grid/fold",
                   **split_meta, snapshot=snapshot, n_users=len(valid), users=valid),
              open(VALID300, "w"))
    json.dump(dict(calls=calls, prompt_tokens=ptok, completion_tokens=ctok, usd=usd,
                   snapshot=snapshot, tokstat=tokstat, n_users=len(grid),
                   duplicates=[dict(user=d[0], kept=d[1], dropped=d[2]) for d in dup]),
              open(USAGE300, "w"), indent=1)
    print(f"[merge] {len(grid)} users -> {GRID300}; {len(valid)} valid -> {VALID300}")
    print(f"[merge] total ${usd:.4f}, {calls} calls; duplicates judged-twice = {len(dup)}")
    for d in dup:
        print(f"    DUP user {d[0]}: kept {d[1]}, dropped {d[2]}")
    if len(grid) != 300:
        print(f"[merge] WARNING: expected 300 users, have {len(grid)}")
    return grid, valid


# ------------------------------------------------------------------ alt split (same convention, alt seed)
def build_alt_split(D, seed):
    rs = np.random.default_rng(seed)
    split = {}
    for u, v in D["rat_by_u"].items():
        its = list(dict(v))
        if len(its) >= 6:
            il = its[:]; rs.shuffle(il)
            split[u] = (set(il[:len(il) // 2]), il[len(il) // 2:])
    return split


def stratified_tags(tags, n):
    """n tags stratified into terciles by answer_rate_prior, deterministic even spacing."""
    order = sorted(range(len(tags)), key=lambda i: (tags[i]["answer_rate_prior"], tags[i]["tagId"]))
    thirds = np.array_split(order, 3)
    per = n // 3
    pick = []
    for arr in thirds:
        arr = list(arr)
        if len(arr) <= per:
            sel = arr
        else:
            step = len(arr) / per
            sel = [arr[int(i * step)] for i in range(per)]
        pick.extend(sel)
    pick = sorted(set(pick))
    for i in order:                          # top up to exactly n, preserving order
        if len(pick) >= n:
            break
        if i not in pick:
            pick.append(i)
    pick = sorted(set(pick))[:n]
    return [tags[i] for i in pick]


def stratified_items(D, top800, n):
    """n items from top800 stratified into terciles by popularity cnt, deterministic even spacing."""
    order = sorted(top800, key=lambda j: (D["cnt"][j], j))
    thirds = np.array_split(order, 3)
    per = n // 3
    pick = []
    for arr in thirds:
        arr = list(arr)
        if len(arr) <= per:
            sel = arr
        else:
            step = len(arr) / per
            sel = [arr[int(i * step)] for i in range(per)]
        pick.extend(sel)
    pick = sorted(set(int(j) for j in pick))
    for j in order:                          # top up to exactly n
        if len(pick) >= n:
            break
        if int(j) not in pick:
            pick.append(int(j))
    pick = sorted(set(int(j) for j in pick))[:n]
    return pick


def ss_judge(u, kn_set, rat, D, ss_tags, ss_items, tokstat):
    """Judge the reduced battery for ONE user under a given known-half. Returns
    {concept:{tagId:knowledge}, item:{j:knowledge}} using the EXACT SB prompt/parse."""
    kn = sorted(kn_set)
    plines, capped = SB.profile_lines(D, set(kn), rat)
    prof = "; ".join(plines) if plines else "(no ratings shown)"
    if capped:
        prof = "[a star-stratified sample of this viewer's ratings]\n" + prof
    n_shown = len(plines); N_total = len(rat)
    # build the question list (concept tags then items), EXACT compact rendering as SB.run_production
    Q = []
    for t in ss_tags:
        Q.append(("concept", dict(tagId=t["tagId"], tag=t["tag"]), f"{t['question']}"))
    for j in ss_items:
        Q.append(("item", dict(j=int(j)), f"film '{D['title'][j]}'"))
    ans = {}
    for start in range(0, len(Q), SB.BATCH):
        chunk = Q[start:start + SB.BATCH]
        lines = [f"{i}: {qt}" for i, (_, _, qt) in enumerate(chunk)]
        user_msg = (f"Viewer's rating sample (size {n_shown} of {N_total} total; they have seen many more "
                    f"films than listed):\n{prof}\n\nQuestions:\n" + "\n".join(lines))
        arr = SB.call_and_count(SB.SYS_PROD, user_msg, len(chunk), tokstat)
        for pos, o in enumerate(arr):
            f = SB._row_fields(o)
            if f is None:
                continue
            try:
                li = int(f[0])
            except (ValueError, TypeError):
                li = pos
            if 0 <= li < len(chunk):
                cell, _, _ = SB.parse_compact(o)
                ans[start + li] = cell
    out = {"concept": {}, "item": {}}
    for i, (ch, m, _t) in enumerate(Q):
        c = ans.get(i)
        kn_lab = c.get("knowledge") if (c and "_raw" not in c) else None
        if ch == "concept":
            out["concept"][int(m["tagId"])] = kn_lab
        else:
            out["item"][int(m["j"])] = kn_lab
    return out


def splitsens():
    D = G.load_data()
    split123 = G.build_split(D)
    battery, tags, top800 = SB.load_inputs()
    cohort = load_cohort(split123)
    ss_users = cohort[::(len(cohort) // N_SS_USERS)][:N_SS_USERS]      # deterministic even sample
    ss_tags = stratified_tags(tags, N_SS_TAGS)
    ss_items = stratified_items(D, top800, N_SS_ITEMS)
    print(f"[splitsens] {len(ss_users)} users x seeds {ALT_SEEDS} x ({len(ss_tags)} tags + "
          f"{len(ss_items)} items)", flush=True)

    # seed-123 reference labels from the FROZEN grid300
    grid300 = json.load(open(GRID300))["users"]

    def ref_labels(u):
        rec = grid300[str(u)]
        cmap = {}; imap = {}
        for _i, c in rec["Q"].items():
            o = c.get("ans")
            lab = o.get("knowledge") if (o and "_raw" not in o) else None
            if c["channel"] == "concept":
                cmap[int(c["tagId"])] = lab
            elif c["channel"] == "item":
                imap[int(c["j"])] = lab
        for j, c in rec.get("data", {}).items():
            imap[int(c["j"])] = c.get("knowledge")     # data-filled = know_well
        return cmap, imap

    tokstat = {"completion": 0, "prompt": 0, "questions": 0, "calls": 0}
    base_usd = 0.0
    result = {"users": {}, "alt_seeds": ALT_SEEDS, "n_tags": len(ss_tags), "n_items": len(ss_items),
              "tags": [t["tagId"] for t in ss_tags], "items": ss_items}
    if os.path.exists(SPLITSENS):
        prev = json.load(open(SPLITSENS))
        result["users"] = prev.get("users", {})
        if os.path.exists(SPLITSENS_USAGE):
            base_usd = json.load(open(SPLITSENS_USAGE)).get("usd", 0.0)
    alt_splits = {s: build_alt_split(D, s) for s in ALT_SEEDS}

    for u in ss_users:
        rat = dict(D["rat_by_u"][u])
        cref, iref = ref_labels(u)
        urec = result["users"].get(str(u), {})
        for s in ALT_SEEDS:
            key = f"seed{s}"
            if key in urec:
                continue
            kn_alt, _ho = alt_splits[s][u]
            judged = ss_judge(u, kn_alt, rat, D, ss_tags, ss_items, tokstat)
            urec[key] = judged
            result["users"][str(u)] = urec
            json.dump(result, open(SPLITSENS, "w"))
            cum = base_usd + G._USAGE["usd"]
            json.dump(dict(usd=cum, tokstat=tokstat, snapshot=G._USAGE["snapshot"]),
                      open(SPLITSENS_USAGE, "w"), indent=1)
            print(f"  [ss] u{u} {key}: judged {len(judged['concept'])}c+{len(judged['item'])}i "
                  f"| cum ${cum:.4f}", flush=True)
        # stash the seed-123 reference for this user (labels only)
        urec["ref123"] = {"concept": {str(k): v for k, v in cref.items()},
                          "item": {str(k): v for k, v in iref.items()}}
        result["users"][str(u)] = urec
        json.dump(result, open(SPLITSENS, "w"))
    print(f"[splitsens] done; cum ${base_usd + G._USAGE['usd']:.4f}", flush=True)


# ------------------------------------------------------------------ kappa
def cohen_kappa(pairs, cats):
    """pairs: list of (a,b) categorical labels; cats: category list. Returns kappa or nan."""
    if len(pairs) < 2:
        return float("nan")
    idx = {c: i for i, c in enumerate(cats)}
    n = len(pairs); K = len(cats)
    conf = np.zeros((K, K))
    for a, b in pairs:
        conf[idx[a], idx[b]] += 1
    po = np.trace(conf) / n
    ra = conf.sum(1) / n; rb = conf.sum(0) / n
    pe = float(np.sum(ra * rb))
    if pe >= 1.0:
        return float("nan")
    return float((po - pe) / (1 - pe))


# ------------------------------------------------------------------ report
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def report():
    D = G.load_data()
    grid = json.load(open(GRID300))["users"]
    valid = json.load(open(VALID300))["users"]
    usage = json.load(open(USAGE300)) if os.path.exists(USAGE300) else {}
    users = sorted(int(u) for u in grid)
    L = []; P = L.append

    # collect channel cells
    cells = {"concept": [], "attribute": [], "item": []}
    n_data = 0
    for us in users:
        rec = grid[str(us)]
        for _i, c in rec["Q"].items():
            cells[c["channel"]].append((us, c, c.get("ans")))
        n_data += len(rec.get("data", {}))

    snapshot = usage.get("snapshot", G._USAGE.get("snapshot") or G.MODEL)
    P("# Answerer v1.0 -- FINAL 300-USER RUN (validation & freeze report)")
    P("")
    P(f"Judge `{snapshot}` temp {G.TEMPERATURE}, answerer split seed {SPLIT_SEED}. {len(users)} study "
      f"users (answerability-gate cohort). Judging machinery BYTE-IDENTICAL to the validated small batch "
      f"(scripts/answerer_smallbatch.py); 10 small-batch users absorbed from answerer_v1_grid10.json, "
      f"not re-judged. Grid `{GRID300}`; validation `{VALID300}` (QUARANTINED).")
    P("")

    # ---- (a) cost ----
    P("## (a) Cost (exact, from merged sidecars)")
    P("")
    usd = usage.get("usd", 0.0); ncalls = usage.get("calls", 0)
    ss_usd = json.load(open(SPLITSENS_USAGE)).get("usd", 0.0) if os.path.exists(SPLITSENS_USAGE) else 0.0
    P(f"- main grid: {ncalls} calls, {usage.get('prompt_tokens')} prompt + "
      f"{usage.get('completion_tokens')} completion tokens, **${usd:.4f}** "
      f"(rates ${G.PRICE_IN_PER_M}/M in, ${G.PRICE_OUT_PER_M}/M out)")
    P(f"- per-user ${usd/max(len(users),1):.4f}")
    P(f"- split-sensitivity pass: **${ss_usd:.4f}**")
    P(f"- **GRAND TOTAL = ${usd + ss_usd:.4f}** (budget cap ${BUDGET_CAP:.0f})")
    if usage.get("duplicates"):
        P(f"- users judged twice across shards (kept first): {len(usage['duplicates'])} "
          f"-> {usage['duplicates']}")
    else:
        P(f"- users judged twice across shards: 0")
    P("")

    # ---- (b) parse / core-contract ----
    P("## (b) Parse / core-contract rate")
    P("")
    P("| channel | cells | returned | core-valid | core% | conf% | hard-invalid |")
    P("|---|--:|--:|--:|--:|--:|--:|")
    tot_cells = tot_core = tot_coerced = 0
    invalid_examples = []
    for ch in ("concept", "attribute", "item"):
        tot = len(cells[ch]); ret = core = conf = hard = coerced = 0
        for (u, m, o) in cells[ch]:
            if o is None:
                continue
            ret += 1
            if "_raw" in o:
                hard += 1
                if len(invalid_examples) < 20:
                    invalid_examples.append((ch, u, m.get("tag") or m.get("entity_id") or m.get("j")))
                continue
            kk = o.get("knowledge")
            ok = kk in KNOW and ((kk == "no_clue" and "stars" not in o) or
                                 (kk != "no_clue" and o.get("value") in VAL))
            core += 1 if ok else 0
            hard += 0 if ok else 1
            conf += 1 if "conf" in o else 0
            coerced += 1 if "coerced_from_k" in o else 0
        tot_cells += tot; tot_core += core; tot_coerced += coerced
        P(f"| {ch} | {tot} | {ret} | {core} | {100*core/max(tot,1):.2f}% | "
          f"{100*conf/max(ret,1):.2f}% | {hard} |")
    P(f"| **data-filled** | {n_data} | {n_data} | {n_data} | 100.00% | 100.00% | 0 |")
    P("")
    P(f"Overall LLM core-contract = **{100*tot_core/max(tot_cells,1):.2f}%** "
      f"({'PASS >=99%' if 100*tot_core/max(tot_cells,1) >= 99 else 'BELOW 99%'}).")
    P(f"Coerced-to-no_clue (k>=1 but no usable rating): {tot_coerced}.")
    if invalid_examples:
        P(f"Residual invalid cells (channel,user,key): {invalid_examples}")
    else:
        P("Residual invalid cells: NONE.")
    P("")

    # ---- (c) validation ----
    P("## (c) Validation (held-out-half items -- QUARANTINED)")
    P("")
    stars_real = []; vpairs = []; liked = 0; nval = 0; n_noclue = 0; n_rows = 0
    per_user_corr = []
    for us in users:
        us_stars = []
        for row in valid.get(str(us), {}).get("rows", []):
            n_rows += 1
            c = row.get("pred")
            if not c or "_raw" in c:
                continue
            if c.get("knowledge") == "no_clue":
                n_noclue += 1; continue
            if "stars" in c:
                stars_real.append((float(c["stars"]), float(row["true_rating"])))
                us_stars.append((float(c["stars"]), float(row["true_rating"])))
                vpairs.append((c["value"], row["true_value"])); nval += 1
                if c.get("value") == "liked":
                    liked += 1
        if len(us_stars) >= 3:
            a = np.array([x for x, _ in us_stars]); b = np.array([y for _, y in us_stars])
            if a.std() > 0 and b.std() > 0:
                per_user_corr.append(float(np.corrcoef(a, b)[0, 1]))
    a = np.array([x for x, _ in stars_real]); b = np.array([y for _, y in stars_real])
    corr = float(np.corrcoef(a, b)[0, 1]) if len(a) > 3 and a.std() > 0 and b.std() > 0 else float("nan")
    mae = float(np.mean(np.abs(a - b))) if len(a) else float("nan")
    vag = SB.agree_stats(vpairs)
    P(f"- n held-out items judged = {n_rows}; non-no_clue with stars = {nval}; no_clue = {n_noclue}")
    P(f"- **pooled corr(predicted stars, true held-out rating) = {corr:.3f}** (small-batch 0.547)")
    P(f"- star MAE = {mae:.3f} stars")
    P(f"- binned exact = {vag.get('exact',float('nan')):.3f} | adjacent = {vag.get('adjacent',float('nan')):.3f}")
    P(f"- **liked-share = {liked/max(nval,1):.3f}** (n={nval})")
    if per_user_corr:
        pc = np.array(per_user_corr)
        P(f"- per-user corr distribution (n_users={len(pc)} with >=3 rated preds): "
          f"mean {pc.mean():.3f}, sd {pc.std():.3f}, min {pc.min():.3f}, "
          f"q25 {np.percentile(pc,25):.3f}, median {np.median(pc):.3f}, q75 {np.percentile(pc,75):.3f}, "
          f"max {pc.max():.3f}")
    lm = collections.Counter(x for x, _ in vpairs); tm = collections.Counter(y for _, y in vpairs)
    nn = max(len(vpairs), 1)
    P(f"- marginals -- PRED: " + ", ".join(f"{v} {lm.get(v,0)/nn:.3f}" for v in VAL) +
      " | TRUE: " + ", ".join(f"{v} {tm.get(v,0)/nn:.3f}" for v in VAL))
    P("")

    # ---- (d) knowledge base rates ----
    P("## (d) Knowledge base rates per channel")
    P("")
    P("| channel | n | no_clue | rough_idea | know_well |")
    P("|---|--:|--:|--:|--:|")
    for ch in ("concept", "attribute", "item"):
        objs = [o for (_, _, o) in cells[ch] if o is not None and "_raw" not in o]
        c = collections.Counter(o.get("knowledge") for o in objs); n = sum(c.values())
        if n:
            P(f"| {ch} | {n} | {c.get('no_clue',0)/n:.3f} | {c.get('rough_idea',0)/n:.3f} | "
              f"{c.get('know_well',0)/n:.3f} |")
    allobj = [o for ch in cells for (_, _, o) in cells[ch] if o is not None and "_raw" not in o]
    ac = collections.Counter(o.get("knowledge") for o in allobj); an = sum(ac.values())
    P("")
    P(f"Overall rough_idea share = {ac.get('rough_idea',0)/max(an,1):.3f} "
      f"({'USED' if ac.get('rough_idea',0)/max(an,1) > 0.02 else 'COLLAPSED'}).")
    P("")

    # popularity/prominence-stratum monotonicity
    P("Item answerability by popularity tier (monotonicity: know_well should rise with popularity):")
    P("")
    P("| tier | n | no_clue | rough_idea | know_well |")
    P("|---|--:|--:|--:|--:|")
    ord_tiers = ["obscure", "moderate", "famous"]
    kw_by_tier = {}
    for tname in ord_tiers:
        objs = [o for (_, m, o) in cells["item"] if o is not None and "_raw" not in o
                and D["tier"][m["j"]] == tname]
        c = collections.Counter(o.get("knowledge") for o in objs); n = sum(c.values())
        if n:
            kw_by_tier[tname] = c.get("know_well", 0) / n
            P(f"| {tname} | {n} | {c.get('no_clue',0)/n:.3f} | {c.get('rough_idea',0)/n:.3f} | "
              f"{c.get('know_well',0)/n:.3f} |")
    mono = all(kw_by_tier.get(ord_tiers[i], 0) <= kw_by_tier.get(ord_tiers[i + 1], 0)
               for i in range(len(ord_tiers) - 1) if ord_tiers[i] in kw_by_tier and ord_tiers[i+1] in kw_by_tier)
    P("")
    P(f"know_well monotone increasing with popularity: {'YES' if mono else 'NO'} "
      f"({', '.join(f'{t}={kw_by_tier.get(t,0):.3f}' for t in ord_tiers)}).")
    P("")

    # attribute by type
    P("Attribute answerability by type (composers/writers expected weakest):")
    P("")
    P("| type | n | no_clue | rough_idea | know_well |")
    P("|---|--:|--:|--:|--:|")
    byt = collections.defaultdict(list)
    for (u, m, o) in cells["attribute"]:
        if o is not None and "_raw" not in o:
            byt[m["atype"]].append(o)
    for t in ("director", "actor", "composer", "writer", "franchise"):
        objs = byt[t]; c = collections.Counter(o.get("knowledge") for o in objs); n = sum(c.values())
        if n:
            P(f"| {t} | {n} | {c.get('no_clue',0)/n:.3f} | {c.get('rough_idea',0)/n:.3f} | "
              f"{c.get('know_well',0)/n:.3f} |")
    P("")

    # ---- (e) degenerate columns at n=300 (AUTHOR DECISION, no prune applied) ----
    P("## (e) Degenerate columns at n=300 (AUTHOR DECISION -- no pruning applied)")
    P("")

    def label_entropy(labs):
        n = len(labs)
        c = collections.Counter(labs)
        return -sum((v / n) * np.log2(v / n) for v in c.values() if v > 0)

    # concept columns
    col = collections.defaultdict(dict)
    for (u, m, o) in cells["concept"]:
        if o is None or "_raw" in o:
            continue
        col[m["tag"]][u] = o.get("knowledge")
    tag_deg = []; tag_allnoclue = []
    tag_ents = []
    for tg, um in col.items():
        labs = list(um.values()); n = len(labs)
        if n < 30:
            continue
        ent = label_entropy(labs); tag_ents.append((tg, ent, n))
        top, tn = collections.Counter(labs).most_common(1)[0]
        if tn / n >= 0.90:
            tag_deg.append((tg, top, tn / n, n))
            if top == "no_clue" and tn == n:
                tag_allnoclue.append(tg)
    byl = collections.Counter(f[1] for f in tag_deg)
    P(f"- TAG columns (n>=30 users): {len(col)} scanned; **{len(tag_deg)} degenerate** "
      f"(>=90% share one label). By label: {dict(byl)}.")
    P(f"- all-no_clue TAG columns (100% no_clue): {len(tag_allnoclue)}. "
      f"Examples: {sorted(tag_allnoclue)[:25]}")
    tag_ents.sort(key=lambda x: x[1])
    P(f"- lowest-entropy TAG columns (bits): "
      f"{[(t, round(e,3)) for t, e, _ in tag_ents[:12]]}")

    # attribute columns
    acol = collections.defaultdict(list)
    ameta = {}
    for (u, m, o) in cells["attribute"]:
        if o is None or "_raw" in o:
            continue
        acol[m["entity_id"]].append(o.get("knowledge")); ameta[m["entity_id"]] = m
    attr_deg = []; attr_allnoclue = []
    for eid, labs in acol.items():
        n = len(labs)
        if n < 30:
            continue
        top, tn = collections.Counter(labs).most_common(1)[0]
        if tn / n >= 0.90:
            attr_deg.append((eid, ameta[eid].get("atype"), top, tn / n))
            if top == "no_clue" and tn == n:
                attr_allnoclue.append((eid, ameta[eid].get("atype")))
    adeg_bytype = collections.Counter(a[1] for a in attr_deg)
    P(f"- ATTRIBUTE columns: {len(acol)} scanned; **{len(attr_deg)} degenerate**. "
      f"By type: {dict(adeg_bytype)}. all-no_clue: {len(attr_allnoclue)}.")

    # item columns (LLM-judged only)
    icol = collections.defaultdict(list)
    for (u, m, o) in cells["item"]:
        if o is None or "_raw" in o:
            continue
        icol[m["j"]].append(o.get("knowledge"))
    item_deg = 0; item_allnoclue = 0
    for j, labs in icol.items():
        n = len(labs)
        if n < 30:
            continue
        top, tn = collections.Counter(labs).most_common(1)[0]
        if tn / n >= 0.90:
            item_deg += 1
            if top == "no_clue" and tn == n:
                item_allnoclue += 1
    P(f"- ITEM columns (LLM-judged, n>=30): {len(icol)} scanned; **{item_deg} degenerate**, "
      f"{item_allnoclue} all-no_clue.")
    P("")
    P("AUTHOR DECISION: no columns pruned. Degeneracy is reported as prune EVIDENCE for a later, "
      "separately-approved decision; the frozen v1.0 environment retains the full battery.")
    P("")

    # ---- (f) judge-consistency vs old grids ----
    P("## (f) Judge-consistency vs prior grids (overlapping cells, kappa)")
    P("")
    # compare grid300 concept+item knowledge labels vs grid10 for the 10 overlap users
    g10 = json.load(open(GRID10))["users"]
    for other_name, other in [("grid10", g10)]:
        cpairs = []; ipairs = []
        for u in other:
            if u not in grid:
                continue
            # index concept by tagId, item by j
            def lab_map(rec):
                cm = {}; im = {}
                for _i, c in rec["Q"].items():
                    o = c.get("ans"); lab = o.get("knowledge") if (o and "_raw" not in o) else None
                    if lab is None:
                        continue
                    if c["channel"] == "concept":
                        cm[int(c["tagId"])] = lab
                    elif c["channel"] == "item":
                        im[int(c["j"])] = lab
                return cm, im
            cA, iA = lab_map(grid[u]); cB, iB = lab_map(other[u])
            for k in set(cA) & set(cB):
                cpairs.append((cA[k], cB[k]))
            for k in set(iA) & set(iB):
                ipairs.append((iA[k], iB[k]))
        kc = cohen_kappa(cpairs, KNOW); ki = cohen_kappa(ipairs, KNOW)
        exact_c = np.mean([a == b for a, b in cpairs]) if cpairs else float("nan")
        exact_i = np.mean([a == b for a, b in ipairs]) if ipairs else float("nan")
        P(f"- vs {other_name} (10 shared users, SAME split/profile -> expect near-identical): "
          f"concept kappa {kc:.3f} (exact {exact_c:.3f}, n={len(cpairs)}); "
          f"item kappa {ki:.3f} (exact {exact_i:.3f}, n={len(ipairs)})")
    P("")

    # ---- (g) split-sensitivity (Hole-5) ----
    P("## (g) Split-sensitivity (Hole-5 closure)")
    P("")
    if os.path.exists(SPLITSENS):
        ss = json.load(open(SPLITSENS))
        P(f"{len([u for u in ss['users']])} users x seeds {ss['alt_seeds']} x "
          f"({ss['n_tags']} stratified tags + {ss['n_items']} stratified items). Alternate known-half "
          f"profiles re-judged; compared to the frozen seed-123 grid on overlapping cells.")
        P("")
        P("| alt seed | channel | kappa | exact | n cells | mean |answer-rate shift| |")
        P("|---|---|--:|--:|--:|--:|")
        for s in ss["alt_seeds"]:
            key = f"seed{s}"
            for ch in ("concept", "item"):
                pairs = []; shifts = []
                for u, urec in ss["users"].items():
                    if key not in urec or "ref123" not in urec:
                        continue
                    alt = urec[key][ch]; ref = urec["ref123"][ch]
                    # answer-rate shift on overlapping labelled cells
                    ov = [k for k in alt if k in ref and alt[k] is not None and ref[k] is not None]
                    if ov:
                        ar_alt = np.mean([alt[k] != "no_clue" for k in ov])
                        ar_ref = np.mean([ref[k] != "no_clue" for k in ov])
                        shifts.append(abs(ar_alt - ar_ref))
                    for k in ov:
                        pairs.append((ref[k], alt[k]))
                kap = cohen_kappa(pairs, KNOW)
                exact = np.mean([a == b for a, b in pairs]) if pairs else float("nan")
                msh = float(np.mean(shifts)) if shifts else float("nan")
                P(f"| {s} | {ch} | {kap:.3f} | {exact:.3f} | {len(pairs)} | {msh:.3f} |")
        P("")
        P("Interpretation: high kappa / low answer-rate shift => judgments are ROBUST to the pinned "
          "answerer split; the seed-123 environment is not an artifact of the one split.")
    else:
        P("(split-sensitivity not yet run -- run `--splitsens`)")
    P("")

    # ---- (h) FREEZE MANIFEST ----
    P("## (h) FREEZE MANIFEST -- answerer v1.0 (frozen environment)")
    P("")
    manifest_files = [
        GRID300, VALID300, USAGE300,
        f"{CACHE}/attr_battery_500.json", f"{CACHE}/tag_questions.json", f"{CACHE}/item_lists.json",
        f"{CACHE}/answerability_answerer_split.json",
        "scripts/answerer_smallbatch.py", "scripts/answerer_fullrun.py",
    ]
    if os.path.exists(SPLITSENS):
        manifest_files.append(SPLITSENS)
    P("| file | sha256 | bytes |")
    P("|---|---|--:|")
    manifest = {}
    for f in manifest_files:
        if os.path.exists(f):
            h = sha256_file(f); sz = os.path.getsize(f)
            manifest[f] = h
            P(f"| `{f}` | `{h}` | {sz} |")
        else:
            P(f"| `{f}` | (MISSING) | - |")
    P("")
    P("Schema: cell = {knowledge in {no_clue,rough_idea,know_well}, stars in [0.5,5.0] (absent iff "
      "no_clue), value in {hated,meh,liked,loved}, conf in [0,1], source in {llm,data}}. Compact wire "
      "row [i,k,h,c]; k=knowledge 0/1/2, h=half-stars 1..10 (0 iff k==0), c=conf tenths 0..10.")
    P("")
    P(f"This is **answerer v1.0** -- the frozen elicitation environment.")

    os.makedirs("experiments", exist_ok=True)
    open(REPORT, "w", encoding="utf-8").write("\n".join(L))
    json.dump(manifest, open(f"{CACHE}/answerer_v1_freeze_manifest.json", "w"), indent=1)
    print(f"[saved] {REPORT} ({len(L)} lines)")
    print("\n==== SUMMARY ====")
    print(f"users {len(users)} | parse core-contract {100*tot_core/max(tot_cells,1):.2f}%")
    print(f"validation corr {corr:.3f} liked-share {liked/max(nval,1):.3f} (n={nval})")
    print(f"total ${usd + ss_usd:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=None)
    ap.add_argument("--nshards", type=int, default=4)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--costcheck", action="store_true")
    ap.add_argument("--splitsens", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.worker is not None:
        run_worker(a.worker, a.nshards)
    elif a.merge:
        merge()
    elif a.costcheck:
        costcheck()
    elif a.splitsens:
        splitsens()
    elif a.report:
        report()
    else:
        print("pass --worker K --nshards N | --merge | --costcheck | --splitsens | --report")
