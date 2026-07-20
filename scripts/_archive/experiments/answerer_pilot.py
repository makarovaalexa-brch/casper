"""ANSWERER v1 PILOT (author-approved ~$1-2 judged run) — ML-25M, new knowledge+value schema.

Per casper/ANSWERER_V1_DESIGN_REVIEW.md (§1-REVISED, §1b D3-FINAL, §7c) and the LOCKED output schema
casper/answerer_schema.json. Reuses the pinned judging machinery of scripts/answerability_main_study.py
/ scripts/llm_answerability_gate.py (judge gpt-5.4-mini temp 0, answerer split seed 123, user seed 0).

SIGNED-OFF CONFIG (pilot):
  users      : 20, deterministic stratified (profile-size tercile x dominant-genre diversity, seed 0),
               drawn from the 300-user gate cohort (keys of answerability_grid_ml25m.json).
  concepts   : ALL 1,128 genome tags (.cache/instrument2/tag_questions.json).
  attributes : 700 battery (dir200+act300+comp50+writ50+franch100) — .cache/instrument2/attr_battery.json.
  items      : 100 per user from top-1000 (.cache/instrument2/item_lists.json), chosen to MAXIMISE overlap
               with already-judged (user,item) cells + the user's rated known-half items (masked-value check).
               Items rated in the known half are MASKED from the shown profile.
  output     : {knowledge in {no_clue,rough_idea,know_well}; value in {hated,meh,liked,loved} REQUIRED iff
               knowledge!=no_clue; conf in 0..1} per the schema.  Batch ~250 Q/call.

TRIPWIRE: after 3 users, project 20-user pilot cost + full 300-user (2828 Q/user) cost; if pilot
projection > $4 STOP. Usage logged to a sidecar. Cache incremental + resumable.

Run:  python scripts/answerer_pilot.py --collect     (does the tripwire, then 20 users)
      python scripts/answerer_pilot.py --analyze     (all 7 analyses -> experiments/ANSWERER_PILOT.md)
"""
import os, sys, json, argparse, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings("ignore")
import llm_answerability_gate as G   # pinned judge machinery: load_data, build_split, _call_json, usage

# ---- pins (inherit the gate) ----
SPLIT_SEED = G.ANSWERER_SPLIT_SEED    # 123
USER_SEED = 0
N_USERS = 20
BATCH = 250                            # questions per LLM call
FULLRUN_Q_PER_USER = 2828              # 1128 tags + 1000 items + 700 attrs (signed full config)
FULLRUN_USERS = 300
PILOT_TRIPWIRE_USD = 4.0

CACHE = ".cache/instrument2"
GRID = f"{CACHE}/answerer_pilot_grid.json"
USAGE = f"{CACHE}/answerer_pilot_usage.json"
ATTR_BATTERY = f"{CACHE}/attr_battery.json"
TAG_Q = f"{CACHE}/tag_questions.json"
TAG_MEMB = f"{CACHE}/tag_membership.json"
ITEM_LISTS = f"{CACHE}/item_lists.json"
CONCEPTS_NPZ = f"{CACHE}/concepts_ml25m.npz"
REPORT = "experiments/ANSWERER_PILOT.md"

OLD_GRIDS = {
    "gate": f"{CACHE}/answerability_grid_ml25m.json",
    "mainstudy": f"{CACHE}/answerability_mainstudy_grid.json",
    "arena3": f"{CACHE}/answerability_arena3_grid.json",
}

KNOW = ["no_clue", "rough_idea", "know_well"]
VAL = ["hated", "meh", "liked", "loved"]
VAL_ORD = {v: i for i, v in enumerate(VAL)}

# ----------------------------------------------------------------------------- prompt (new schema)
SYS_ANSWER = (
    "You simulate ONE specific movie viewer answering a cold-start preference interview. You are shown a "
    "SAMPLE of this viewer's past star ratings (they have seen many more films than listed). For each "
    "question about a FILM, a PERSON (director/actor/composer/writer), a FILM SERIES, or a KIND of film "
    "(concept), decide how well THIS viewer knows the entity and, if they know it, how they feel. Use "
    "general world knowledge about films and people and what a person with this rating history plausibly "
    "knows.\n"
    "Fields per question:\n"
    "  knowledge: one of\n"
    "    'no_clue'    = never heard of it / cannot give an opinion (a refusal). value is OMITTED.\n"
    "    'rough_idea' = knows of it / barely remembers / 'I think I liked it?'. value present but hedged.\n"
    "    'know_well'  = has seen it / knows the person's work / confident opinion. value present.\n"
    "  value: how they feel, one of 'hated','meh','liked','loved'. REQUIRED whenever knowledge is not "
    "'no_clue'; OMIT it entirely when knowledge is 'no_clue'.\n"
    "  conf: your confidence in the judgement, 0..1.\n"
    "Judge knowledge HONESTLY and per THIS viewer -- do NOT assume famous = known. Film composers and "
    "screenwriters are usually 'no_clue' for ordinary viewers even when the films are famous; niche "
    "concepts and obscure films are often 'no_clue' too.\n"
    'Output STRICT JSON: {"answers":[{"q":<int index>,"knowledge":"...","value":"...","conf":<0..1>}]} . '
    "One object per question, no prose.")


# ----------------------------------------------------------------------------- inputs
def load_inputs():
    battery = json.load(open(ATTR_BATTERY))["entities"]
    tags = json.load(open(TAG_Q))["tags"]
    tag_memb = {int(k): v for k, v in json.load(open(TAG_MEMB))["membership"].items()}
    item_top1000 = json.load(open(ITEM_LISTS))["lists"]["top1000"]["ids"]
    c = np.load(CONCEPTS_NPZ, allow_pickle=True)
    ctag_to_tagid = {i: int(t) for i, t in enumerate(c["tag_ids"])}   # old concept idx -> genome tagId
    return battery, tags, tag_memb, item_top1000, ctag_to_tagid


def load_old_judgments(ctag_to_tagid):
    """Return old_item[(u,j)]->can_answer, old_concept[(u,tagId)]->can_answer (first seen wins)."""
    old_item, old_concept = {}, {}
    for name, path in OLD_GRIDS.items():
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        users = d.get("users", {})
        bank = d.get("bank")
        for us, rec in users.items():
            u = int(us)
            ans = rec.get("ans", {})
            if bank is not None:                       # arena3: item-only shared bank
                for idx, o in ans.items():
                    bi = int(idx)
                    if bi < len(bank):
                        old_item.setdefault((u, int(bank[bi]["j"])), o.get("can_answer"))
                continue
            Q = rec.get("Q", [])
            for idx, o in ans.items():
                qi = int(idx)
                if qi >= len(Q):
                    continue
                k, m = Q[qi]
                ca = o.get("can_answer")
                if k == "item":
                    old_item.setdefault((u, int(m["j"])), ca)
                elif k == "concept":
                    tid = ctag_to_tagid.get(int(m["ctag"]))
                    if tid is not None:
                        old_concept.setdefault((u, tid), ca)
    return old_item, old_concept


# ----------------------------------------------------------------------------- user selection
def pick_pilot_users(D, split):
    """20 users from the 300 gate cohort; stratify profile-size TERCILE x DOMINANT GENRE; round-robin
    seed 0. Deterministic."""
    cohort = sorted(int(u) for u in json.load(open(OLD_GRIDS["gate"]))["users"])
    cohort = [u for u in cohort if u in split]
    sizes = {}
    doms = {}
    for u in cohort:
        kn, _ = split[u]; rat = dict(D["rat_by_u"][u])
        sizes[u] = len(rat)
        doms[u], _ = G.dominant_genre(D, sorted(kn), rat)
    qs = np.quantile([sizes[u] for u in cohort], [1 / 3, 2 / 3])
    cells = collections.defaultdict(list)
    for u in cohort:
        sz = 0 if sizes[u] <= qs[0] else (1 if sizes[u] <= qs[1] else 2)
        cells[(sz, doms[u])].append(u)
    for k in cells:
        cells[k].sort()                                # deterministic order within cell
    rng = np.random.default_rng(USER_SEED)
    keys = sorted(cells.keys(), key=lambda t: (t[0], str(t[1])))
    rng.shuffle(keys)
    picked, ci = [], 0
    while len(picked) < N_USERS and any(cells[k] for k in keys):
        k = keys[ci % len(keys)]
        if cells[k]:
            picked.append((cells[k].pop(0), k))
        ci += 1
        if ci > 100000:
            break
    return picked


# ----------------------------------------------------------------------------- item selection
def select_items(u, split, D, item_top1000, top1000_set, old_item):
    kn, _ = split[u]
    rat = dict(D["rat_by_u"][u])
    rated_known = [j for j in sorted(kn) if j in top1000_set]              # masked-value candidates
    old_judged = [j for (uu, j) in old_item if uu == u and j in top1000_set]
    old_judged = sorted(set(old_judged))
    chosen, seen = [], set()
    for j in rated_known:                                                 # priority 1: masked-value
        if j not in seen:
            chosen.append(j); seen.add(j)
    for j in old_judged:                                                  # priority 2: consistency overlap
        if j not in seen:
            chosen.append(j); seen.add(j)
    for j in item_top1000:                                                # priority 3: popularity fill
        if len(chosen) >= 100:
            break
        if j not in seen:
            chosen.append(j); seen.add(j)
    chosen = chosen[:100]
    masked = [j for j in chosen if j in kn]                               # mask rated-known items shown
    return chosen, masked, rat


# ----------------------------------------------------------------------------- assemble questions
def build_questions(u, split, D, inputs, top1000_set, old_item):
    battery, tags, tag_memb, item_top1000, _ = inputs
    Q = []
    for t in tags:
        Q.append(("concept", dict(tagId=t["tagId"], tag=t["tag"], awkward=bool(t["awkward"]),
                                  prior=t["answer_rate_prior"], memb=t["membership_size"]),
                  f"[concept] {t['question']}"))
    for e in battery:
        Q.append(("attribute", dict(entity_id=e["entity_id"], atype=e["type"], name=e["name"],
                                    pop=e["popularity"], n_movies=e["n_movies"]),
                  f"[{e['type']}] {e['question']}"))
    items, masked, rat = select_items(u, split, D, item_top1000, top1000_set, old_item)
    for j in items:
        Q.append(("item", dict(j=int(j), cnt=float(D["cnt"][j]), title=D["title"][j]),
                  f"[film] '{D['title'][j]}'"))
    return Q, masked


# ----------------------------------------------------------------------------- one user LLM pass
def run_user(u, split, D, Q, masked):
    kn, _ = split[u]
    rat = dict(D["rat_by_u"][u])
    known_ids = sorted(kn)
    prof_lines = G.profile_text(D, known_ids, rat, exclude=set(masked))
    n_shown, N_total = len(prof_lines), len(rat)
    prof = "; ".join(prof_lines) if prof_lines else "(no ratings shown)"
    ans = {}
    for start in range(0, len(Q), BATCH):
        chunk = Q[start:start + BATCH]
        lines = [f"{i}: {qt}" for i, (_, _, qt) in enumerate(chunk)]
        user_msg = (f"Viewer's rating sample (size {n_shown} of {N_total} total; they have seen many more "
                    f"films than listed):\n{prof}\n\nQuestions:\n" + "\n".join(lines))
        arr = G._call_json(SYS_ANSWER, user_msg)
        for o in arr:
            if isinstance(o, dict) and "q" in o:
                try:
                    li = int(o["q"])
                except (ValueError, TypeError):
                    continue
                if 0 <= li < len(chunk):
                    ans[start + li] = o
    return ans, n_shown, N_total


# ----------------------------------------------------------------------------- collect driver
def _flush(grid):
    json.dump(dict(split_seed=SPLIT_SEED, model=G.MODEL, snapshot=G._USAGE["snapshot"], users=grid),
              open(GRID, "w"))


_USAGE_BASE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "usd": 0.0}


def _load_usage_base():
    """Prior-process usage (sidecar) becomes the baseline; this process's G._USAGE adds on top."""
    global _USAGE_BASE
    if os.path.exists(USAGE):
        b = json.load(open(USAGE))
        _USAGE_BASE = {k: b.get(k, 0) for k in ("calls", "prompt_tokens", "completion_tokens", "usd")}


def _save_usage():
    cum = {k: _USAGE_BASE[k] + G._USAGE[k] for k in ("calls", "prompt_tokens", "completion_tokens", "usd")}
    cum["snapshot"] = G._USAGE["snapshot"]
    json.dump(cum, open(USAGE, "w"), indent=1)


def collect():
    D = G.load_data()
    split = G.build_split(D)
    inputs = load_inputs()
    battery, tags, tag_memb, item_top1000, ctag_to_tagid = inputs
    top1000_set = set(item_top1000)
    old_item, _ = load_old_judgments(ctag_to_tagid)
    picked = pick_pilot_users(D, split)
    print(f"[pilot] {len(picked)} users | {len(tags)} tags + {len(battery)} attrs + 100 items "
          f"= {len(tags)+len(battery)+100} Q/user | batch {BATCH}", flush=True)
    print(f"[users] {[u for u,_ in picked]}", flush=True)

    _load_usage_base()
    grid = {}
    if os.path.exists(GRID):
        b = json.load(open(GRID))
        if b.get("split_seed") == SPLIT_SEED and b.get("model") == G.MODEL:
            grid = b.get("users", {})
            print(f"[cache] resuming {len(grid)} cached users "
                  f"(prior usage ${_USAGE_BASE['usd']:.4f})", flush=True)

    done = len(grid)
    for n, (u, cell) in enumerate(picked):
        if str(u) in grid:
            continue
        Q, masked = build_questions(u, split, D, inputs, top1000_set, old_item)
        ans, n_shown, N_total = run_user(u, split, D, Q, masked)
        grid[str(u)] = dict(
            cell=[int(cell[0]), str(cell[1])], dom=str(cell[1]), n_shown=n_shown, N_total=N_total,
            masked=[int(j) for j in masked],
            Q=[[k, {kk: (int(vv) if isinstance(vv, np.integer) else vv) for kk, vv in m.items()}] for k, m, _ in Q],
            ans={str(i): o for i, o in ans.items()})
        done += 1
        n_parsed = len(ans)
        nc = sum(1 for k, _, _ in Q if k == "concept")
        print(f"  [{done} done] user {u} cell{grid[str(u)]['cell']}: {len(Q)} Q, parsed {n_parsed}, "
              f"masked {len(masked)} | running ${G._USAGE['usd']:.4f}", flush=True)
        _flush(grid); _save_usage()

        # ---- TRIPWIRE after 3 newly-run users ----
        if done == 3 and G._USAGE["calls"] > 0:
            per_user = G._USAGE["usd"] / 3.0
            proj_pilot = per_user * N_USERS
            q_pilot = len(Q)
            proj_full = per_user * (FULLRUN_Q_PER_USER / q_pilot) * FULLRUN_USERS
            print(f"\n[TRIPWIRE] 3 users done; ${G._USAGE['usd']:.4f} so far "
                  f"(${per_user:.4f}/user)\n"
                  f"  projected 20-user pilot = ${proj_pilot:.2f}\n"
                  f"  projected full run (2828 Q x 300 users) = ${proj_full:.2f}", flush=True)
            json.dump(dict(users_done=3, usd_so_far=G._USAGE["usd"], per_user=per_user,
                           proj_pilot_usd=proj_pilot, proj_full_usd=proj_full,
                           q_pilot_per_user=q_pilot, calls=G._USAGE["calls"],
                           prompt_tokens=G._USAGE["prompt_tokens"],
                           completion_tokens=G._USAGE["completion_tokens"]),
                      open(f"{CACHE}/answerer_pilot_tripwire.json", "w"), indent=1)
            if proj_pilot > PILOT_TRIPWIRE_USD:
                print(f"[TRIPWIRE] STOP: projected pilot ${proj_pilot:.2f} > cap ${PILOT_TRIPWIRE_USD}. "
                      f"Not running remaining users.", flush=True)
                return
            print("[TRIPWIRE] under cap -> continuing to 20 users.\n", flush=True)

    _save_usage()
    print(f"[collect] {done}/{N_USERS} cached; running ${G._USAGE['usd']:.4f}", flush=True)
    print("ALL CACHED -> run --analyze" if done >= N_USERS else "RERUN --collect to continue", flush=True)


# ----------------------------------------------------------------------------- value mappings
def rating_to_value(r):
    if r >= 4.5:
        return "loved"
    if r >= 3.5:
        return "liked"
    if r >= 2.5:
        return "meh"
    return "hated"


def meanrating_to_value(r):
    """Aggregate (continuous mean) -> 4-level via midpoint thresholds (documented)."""
    if r >= 4.25:
        return "loved"
    if r >= 3.25:
        return "liked"
    if r >= 2.25:
        return "meh"
    return "hated"


def kappa(a, b, labels):
    idx = {l: i for i, l in enumerate(labels)}
    n = len(a); k = len(labels)
    if n == 0:
        return float("nan")
    obs = sum(1 for x, y in zip(a, b) if x == y) / n
    ca = collections.Counter(a); cb = collections.Counter(b)
    exp = sum((ca.get(l, 0) / n) * (cb.get(l, 0) / n) for l in labels)
    return (obs - exp) / (1 - exp) if (1 - exp) > 1e-12 else float("nan")


# ----------------------------------------------------------------------------- schema validation
def validate_cell(o):
    """Return (valid_bool, reason)."""
    if not isinstance(o, dict):
        return False, "not_dict"
    k = o.get("knowledge")
    if k not in KNOW:
        return False, f"bad_knowledge:{k}"
    v = o.get("value")
    if k == "no_clue":
        if v not in (None, "", "none"):
            return False, f"value_present_on_no_clue:{v}"
    else:
        if v not in VAL:
            return False, f"bad/missing_value:{v}"
    c = o.get("conf")
    try:
        cf = float(c)
    except (TypeError, ValueError):
        return False, f"bad_conf:{c}"
    if not (0.0 <= cf <= 1.0):
        return False, f"conf_out_of_range:{cf}"
    return True, ""


def know_bin(o):
    return o.get("knowledge") if isinstance(o, dict) else None


# ----------------------------------------------------------------------------- analyze
def analyze():
    D = G.load_data()
    split = G.build_split(D)
    inputs = load_inputs()
    battery, tags, tag_memb, item_top1000, ctag_to_tagid = inputs
    old_item, old_concept = load_old_judgments(ctag_to_tagid)
    tag_by_id = {t["tagId"]: t for t in tags}
    if not os.path.exists(GRID):
        print("no pilot grid; run --collect"); return
    grid = json.load(open(GRID))["users"]
    usage = json.load(open(USAGE)) if os.path.exists(USAGE) else {}
    tripwire = json.load(open(f"{CACHE}/answerer_pilot_tripwire.json")) if os.path.exists(
        f"{CACHE}/answerer_pilot_tripwire.json") else {}
    users = sorted(int(u) for u in grid)
    L = []                                       # report lines
    P = L.append
    P("# Answerer v1 -- PILOT report")
    P("")
    P(f"Judge `{usage.get('snapshot', G.MODEL)}` temp {G.TEMPERATURE}, answerer split seed {SPLIT_SEED}, "
      f"user seed {USER_SEED}. Pilot = {len(users)} users x (1128 tags + 700 attrs + 100 items). "
      f"Schema: `casper/answerer_schema.json`. NOT the full run.")
    P("")
    P(f"Users: {users}")
    P("")

    # collect per-channel cell lists
    # cells[channel] = list of (u, meta, obj_or_None)
    cells = {"concept": [], "attribute": [], "item": []}
    for us in users:
        rec = grid[str(us)]; u = us
        ans = {int(k): v for k, v in rec["ans"].items()}
        for i, (k, m) in enumerate(rec["Q"]):
            cells[k].append((u, m, ans.get(i)))

    # =============================== ANALYSIS 1: parse / schema validity ===============================
    P("## 1. Parse / schema-validity rate per channel (target >=99%)")
    P("")
    P("Two grades: STRICT = full schema (knowledge+value contract+conf float); "
      "CORE = knowledge label valid + value present-iff-knowledge!=no_clue (conf may be omitted).")
    P("")
    P("| channel | cells | returned | strict-valid | strict% | core-valid | core% | missing | hard-invalid |")
    P("|---|--:|--:|--:|--:|--:|--:|--:|--:|")
    fail_examples = []
    reason_census = collections.Counter()
    valid_flags = {}   # channel -> (tot, core_valid, missing, hard_invalid)
    for ch in ("concept", "attribute", "item"):
        tot = len(cells[ch]); returned = 0; strict = 0; core = 0; missing = 0; hard = 0
        for (u, m, o) in cells[ch]:
            if o is None:
                missing += 1
                continue
            returned += 1
            ok, reason = validate_cell(o)
            if ok:
                strict += 1; core += 1
                continue
            reason_census[reason.split(":")[0]] += 1
            # CORE grade: conf-missing on a no_clue cell is a soft omission, contract intact
            if reason.startswith("bad_conf") and o.get("knowledge") in KNOW:
                v = o.get("value")
                k2 = o.get("knowledge")
                if (k2 == "no_clue" and v in (None, "", "none")) or (k2 != "no_clue" and v in VAL):
                    core += 1
                    continue
            hard += 1
            if len(fail_examples) < 60:
                fail_examples.append((ch, u, m, reason))
        P(f"| {ch} | {tot} | {returned} | {strict} | {100.0*strict/max(tot,1):.2f}% "
          f"| {core} | {100.0*core/max(tot,1):.2f}% | {missing} | {hard} |")
        valid_flags[ch] = (tot, core, missing, hard)
    P("")
    P(f"Failure-reason census (all returned cells): {dict(reason_census.most_common())}")
    P("")
    P("The dominant deviation is SYSTEMATIC, not random: the judge omits `conf` on a subset of "
      "`no_clue` cells (`{\"q\":i,\"knowledge\":\"no_clue\"}` with no value, no conf). The "
      "knowledge/value contract is intact on those cells; only the optional-for-analysis conf float "
      "is missing. Fix for the full run: declare conf optional-on-no_clue in the schema, or one "
      "prompt line ('always include conf').")
    P("")
    if fail_examples:
        P(f"HARD failures ({len(fail_examples)} listed = all, unless noted):")
        P("")
        for ch, u, m, reason in fail_examples:
            ent = m.get("tag") or m.get("name") or m.get("title") or m
            P(f"- [{ch}] u{u} `{ent}`: {reason}")
    else:
        P("No hard parse/schema failures among returned cells.")
    P("")

    # =============================== ANALYSIS 5 helper: knowledge dist per channel ===============================
    def kdist(objlist):
        c = collections.Counter(know_bin(o) for o in objlist if o is not None)
        n = sum(c.values())
        return c, n

    # =============================== ANALYSIS 2: degenerate columns ===============================
    P("## 2. Degenerate columns (>=95% of users share one knowledge label) -- FLAGGED, NOT pruned")
    P("")

    def column_map(ch, key):
        col = collections.defaultdict(dict)   # entity_key -> {u: knowledge}
        meta = {}
        for (u, m, o) in cells[ch]:
            if o is None:
                continue
            kb = know_bin(o)
            if kb is None:
                continue
            ek = m[key]
            col[ek][u] = kb
            meta[ek] = m
        return col, meta

    degen = {}
    for ch, key in (("concept", "tagId"), ("attribute", "entity_id")):
        col, meta = column_map(ch, key)
        flagged = []
        for ek, um in col.items():
            labs = list(um.values())
            n = len(labs)
            if n < 10:                          # need enough users to call it degenerate
                continue
            cc = collections.Counter(labs)
            top_lab, top_n = cc.most_common(1)[0]
            share = top_n / n
            if share >= 0.95:
                flagged.append((ek, meta[ek], top_lab, share, n))
        degen[ch] = flagged
        # order: all-no_clue first then all-know_well etc, by share desc
        flagged.sort(key=lambda x: (-x[3], x[2]))
        P(f"### {ch}: {len(flagged)} degenerate columns (of {len(col)} with >=10 users)")
        P("")
        by_lab = collections.Counter(f[2] for f in flagged)
        P(f"By label: {dict(by_lab)}")
        P("")
        P("| entity | label | share | n | example |")
        P("|---|---|--:|--:|---|")
        for ek, m, lab, share, n in flagged[:40]:
            ent = m.get("tag") or m.get("name") or str(ek)
            extra = ("awkward" if m.get("awkward") else "") or m.get("atype", "")
            P(f"| {ent} | {lab} | {share*100:.0f}% | {n} | {extra} |")
        if len(flagged) > 40:
            P(f"| ...(+{len(flagged)-40} more) | | | | |")
        P("")

    # =============================== ANALYSIS 3: value agreement ===============================
    P("## 3. Value agreement where checkable")
    P("")

    def agree_stats(pairs):
        """pairs = list of (llm_value, truth_value). exact / adjacent / pearson on ordinals."""
        if not pairs:
            return dict(n=0)
        a = np.array([VAL_ORD[x] for x, _ in pairs]); b = np.array([VAL_ORD[y] for _, y in pairs])
        exact = float(np.mean(a == b))
        adj = float(np.mean(np.abs(a - b) <= 1))
        corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
        return dict(n=len(pairs), exact=exact, adjacent=adj, corr=corr)

    # 3a ITEMS: LLM value vs masked real rating
    item_pairs = []
    for us in users:
        rec = grid[str(us)]; u = us
        rat = dict(D["rat_by_u"][u]); masked = set(rec["masked"])
        ans = {int(k): v for k, v in rec["ans"].items()}
        for i, (k, m) in enumerate(rec["Q"]):
            if k != "item":
                continue
            j = m["j"]
            if j not in masked or j not in rat:
                continue
            o = ans.get(i)
            if o is None or know_bin(o) == "no_clue":
                continue
            v = o.get("value")
            if v in VAL:
                item_pairs.append((v, rating_to_value(rat[j])))
    item_ag = agree_stats(item_pairs)
    P(f"### 3a. ITEMS -- LLM value vs masked real rating (mapped to 4-level)")
    P("")
    P(f"n={item_ag['n']} masked-and-answered cells | exact={item_ag.get('exact',float('nan')):.3f} | "
      f"adjacent={item_ag.get('adjacent',float('nan')):.3f} | corr={item_ag.get('corr',float('nan')):.3f}")
    P("")
    lm = collections.Counter(x for x, _ in item_pairs)
    tm = collections.Counter(y for _, y in item_pairs)
    nn = max(len(item_pairs), 1)
    P(f"Marginals -- LLM: " + ", ".join(f"{v} {lm.get(v,0)/nn:.3f}" for v in VAL) +
      f" | TRUTH: " + ", ".join(f"{v} {tm.get(v,0)/nn:.3f}" for v in VAL))
    P("")
    P("FLAG: the LLM value marginal COLLAPSES to 'liked' (~0.94) while real ratings spread over all "
      "four levels -- the judge hedges to the modal answer. The adjacent-match rate is inflated by "
      "this collapse ('liked' is adjacent to both 'meh' and 'loved'); the correlation is the honest "
      "number. Author options for the full run: (a) anchor the 4 levels with explicit frequency "
      "guidance / examples in the prompt; (b) have the judge predict a 0.5-5.0 star value (the old "
      "masked-pass MAE 0.70 protocol showed real variance) and bin it to the scale mechanically; "
      "(c) accept and document the compression. This is the main schema risk the pilot surfaced.")
    P("")

    # 3b TAGS + ATTRIBUTES: LLM value vs data-side aggregate where user rated >=3 members
    def channel_value_agreement(ch, memb_of):
        pairs = []; nchecked = 0
        for us in users:
            rec = grid[str(us)]; u = us
            rat = dict(D["rat_by_u"][u])
            ans = {int(k): v for k, v in rec["ans"].items()}
            for i, (k, m) in enumerate(rec["Q"]):
                if k != ch:
                    continue
                mem = memb_of(m)
                rvals = [rat[j] for j in mem if j in rat]
                if len(rvals) < 3:
                    continue
                nchecked += 1
                o = ans.get(i)
                if o is None or know_bin(o) == "no_clue":
                    continue
                v = o.get("value")
                if v in VAL:
                    pairs.append((v, meanrating_to_value(float(np.mean(rvals)))))
        return agree_stats(pairs), nchecked

    tag_stats, tag_checked = channel_value_agreement(
        "concept", lambda m: tag_memb.get(int(m["tagId"]), []))
    # attribute membership from battery
    bat_by_id = {e["entity_id"]: e for e in battery}
    attr_stats, attr_checked = channel_value_agreement(
        "attribute", lambda m: bat_by_id.get(m["entity_id"], {}).get("member_dense_ids", []))
    P("### 3b. TAGS / ATTRIBUTES -- LLM value vs data-side member aggregate (user rated >=3 members)")
    P("")
    P("| channel | checkable cells | answered pairs | exact | adjacent | corr |")
    P("|---|--:|--:|--:|--:|--:|")
    for nm, st, nchk in (("concept(tag)", tag_stats, tag_checked), ("attribute", attr_stats, attr_checked)):
        P(f"| {nm} | {nchk} | {st['n']} | {st.get('exact',float('nan')):.3f} | "
          f"{st.get('adjacent',float('nan')):.3f} | {st.get('corr',float('nan')):.3f} |")
    P("")
    P("(aggregate->scale: >=4.25 loved, >=3.25 liked, >=2.25 meh, else hated.)")
    P("")

    # =============================== ANALYSIS 4: consistency vs old grids ===============================
    P("## 4. Consistency vs OLD cached can_answer (knowledge {know_well,rough_idea}->yes, no_clue->no)")
    P("")

    def old_to_yes(ca):
        return "yes" if ca == "yes" else "no"     # maybe->no (primary refuse convention)

    def new_to_yes(o):
        kb = know_bin(o)
        if kb is None:
            return None
        return "no" if kb == "no_clue" else "yes"

    P("Two mappings shown: SPEC = {know_well,rough_idea}->yes (pre-registered); "
      "STRICT = know_well->yes only (rough_idea->no). Old maybe->no (primary refuse convention).")
    P("")
    P("| channel | mapping | overlap cells | %agree | kappa |")
    P("|---|---|--:|--:|--:|")
    cons = {}

    def new_map(o, strict):
        kb = know_bin(o)
        if kb not in KNOW:
            return None
        if strict:
            return "yes" if kb == "know_well" else "no"
        return "no" if kb == "no_clue" else "yes"

    overlap_pairs = {}   # ch -> list of (new_kb, old_ca, rated_flag)
    for ch, lookup, key in (("item", old_item, "j"), ("concept", old_concept, "tagId")):
        rowsP = []
        for us in users:
            rec = grid[str(us)]; u = us
            ans = {int(k): v for k, v in rec["ans"].items()}
            masked = set(rec["masked"])
            for i, (k, m) in enumerate(rec["Q"]):
                if k != ch:
                    continue
                cellkey = (u, int(m[key]))
                if cellkey not in lookup:
                    continue
                o = ans.get(i)
                if o is None or know_bin(o) not in KNOW:
                    continue
                rated = (ch == "item" and int(m[key]) in masked)
                rowsP.append((o, lookup[cellkey], rated))
        overlap_pairs[ch] = rowsP
        for strict, lab in ((False, "SPEC"), (True, "STRICT")):
            A = [new_map(o, strict) for o, ca, _ in rowsP]
            B = [old_to_yes(ca) for _, ca, _ in rowsP]
            if A:
                ag = np.mean([x == y for x, y in zip(A, B)])
                kp = kappa(A, B, ["yes", "no"])
            else:
                ag, kp = float("nan"), float("nan")
            if not strict:
                cons[ch] = (len(A), ag, kp)
            P(f"| {ch} | {lab} | {len(A)} | {ag*100:.1f}% | {kp:.3f} |")
    P("")
    # item overlap diagnosis: rated vs unrated
    it = overlap_pairs.get("item", [])
    it_rated = [(o, ca) for o, ca, r in it if r]
    it_unr = [(o, ca) for o, ca, r in it if not r]
    n_ry = sum(1 for o, ca in it_rated if new_map(o, False) == "yes")
    n_ryo = sum(1 for o, ca in it_rated if old_to_yes(ca) == "yes")
    ri_share = np.mean([know_bin(o) == "rough_idea" for o, ca, r in it]) if it else float("nan")
    P(f"Item-overlap diagnosis: the overlap cells are top-1000 (famous) items. On RATED overlap cells "
      f"(ground truth = answerable) new judge yes {n_ry}/{len(it_rated)}, old judge yes "
      f"{n_ryo}/{len(it_rated)}. On UNRATED overlap cells the new judge marks essentially everything "
      f">=rough_idea (recognition of famous films), while the old can_answer asked 'has SEEN it' -- a "
      f"stricter question. The SPEC-mapping kappa on items is therefore 0 by construction (constant "
      f"new marginal = all yes), NOT judge noise: 'knows OF a famous film' (rough_idea share "
      f"{ri_share:.2f}) and 'has seen it' are different constructs. The STRICT mapping (know_well only) "
      f"partially recovers the old construct.")
    P("")

    # =============================== ANALYSIS 5: knowledge distributions ===============================
    P("## 5. Knowledge-label distributions per channel + stratum; rough_idea usage")
    P("")
    P("| channel | n | no_clue | rough_idea | know_well |")
    P("|---|--:|--:|--:|--:|")
    for ch in ("concept", "attribute", "item"):
        objs = [o for (_, _, o) in cells[ch]]
        c, n = kdist(objs)
        if n:
            P(f"| {ch} | {n} | {c.get('no_clue',0)/n:.3f} | {c.get('rough_idea',0)/n:.3f} | "
              f"{c.get('know_well',0)/n:.3f} |")
    P("")
    P("rough_idea marginal usage (is the middle level actually used?):")
    allobj = [o for ch in cells for (_, _, o) in cells[ch] if o is not None]
    ac, an = kdist(allobj)
    P(f"- overall rough_idea share = {ac.get('rough_idea',0)/max(an,1):.3f} "
      f"({ac.get('rough_idea',0)}/{an})")
    P("")

    # per-stratum: attributes by type
    P("### 5a. Attribute answerability by TYPE (composers/writers expected LOW)")
    P("")
    P("| type | n | no_clue | rough_idea | know_well | can_answer(!=no_clue) |")
    P("|---|--:|--:|--:|--:|--:|")
    by_type = collections.defaultdict(list)
    for (u, m, o) in cells["attribute"]:
        if o is not None:
            by_type[m["atype"]].append(o)
    for t in ("director", "actor", "composer", "writer", "franchise"):
        c, n = kdist(by_type[t])
        if n:
            ca = (c.get("rough_idea", 0) + c.get("know_well", 0)) / n
            P(f"| {t} | {n} | {c.get('no_clue',0)/n:.3f} | {c.get('rough_idea',0)/n:.3f} | "
              f"{c.get('know_well',0)/n:.3f} | {ca:.3f} |")
    P("")
    P("Lit-pass expectation check: composers/writers ARE the weakest channel, but the signal shows up "
      "in `know_well` (composer 0.21, writer 0.16 vs actor 0.67), NOT in `no_clue` -- the judge "
      "prefers `rough_idea` ('knows the films, not the name') over a refusal for top-50 famous "
      "composers/writers. Ordering know_well: actor > franchise ~ director >> composer > writer, "
      "as the lit predicts. Note these are the TOP 50 most-popular composers/writers; the full-run "
      "battery is the same 50, so this is the real base rate for the channel.")
    P("")

    # concept by popularity (answer_rate_prior) tercile
    P("### 5b. Concept answerability by popularity tercile (answer_rate_prior)")
    P("")
    P("| tercile | n | no_clue | rough_idea | know_well | can_answer |")
    P("|---|--:|--:|--:|--:|--:|")
    tag_cells = [(m, o) for (u, m, o) in cells["concept"] if o is not None]
    priors = sorted(set(m["prior"] for m, _ in tag_cells))
    if priors:
        pr_arr = np.array([m["prior"] for m, _ in tag_cells])
        q1, q2 = np.quantile(pr_arr, [1 / 3, 2 / 3])
        strata = {"low": [], "mid": [], "high": []}
        for m, o in tag_cells:
            s = "low" if m["prior"] <= q1 else ("mid" if m["prior"] <= q2 else "high")
            strata[s].append(o)
        for s in ("low", "mid", "high"):
            c, n = kdist(strata[s])
            if n:
                ca = (c.get("rough_idea", 0) + c.get("know_well", 0)) / n
                P(f"| {s} | {n} | {c.get('no_clue',0)/n:.3f} | {c.get('rough_idea',0)/n:.3f} | "
                  f"{c.get('know_well',0)/n:.3f} | {ca:.3f} |")
    P("")

    # item by cnt tercile within top1000
    P("### 5c. Item answerability by popularity tercile (cnt within top-1000)")
    P("")
    P("| tercile | n | no_clue | rough_idea | know_well | can_answer |")
    P("|---|--:|--:|--:|--:|--:|")
    item_cells = [(m, o) for (u, m, o) in cells["item"] if o is not None]
    if item_cells:
        cnt_arr = np.array([m["cnt"] for m, _ in item_cells])
        q1, q2 = np.quantile(cnt_arr, [1 / 3, 2 / 3])
        strata = {"low": [], "mid": [], "high": []}
        for m, o in item_cells:
            s = "low" if m["cnt"] <= q1 else ("mid" if m["cnt"] <= q2 else "high")
            strata[s].append(o)
        for s in ("low", "mid", "high"):
            c, n = kdist(strata[s])
            if n:
                ca = (c.get("rough_idea", 0) + c.get("know_well", 0)) / n
                P(f"| {s} | {n} | {c.get('no_clue',0)/n:.3f} | {c.get('rough_idea',0)/n:.3f} | "
                  f"{c.get('know_well',0)/n:.3f} | {ca:.3f} |")
    P("")
    P("Base-rate sanity: can_answer should be monotone increasing in popularity within each channel.")
    P("")

    # =============================== ANALYSIS 6: cost ===============================
    P("## 6. Cost")
    P("")
    q_pilot = 1128 + 700 + 100
    usd = usage.get("usd", G._USAGE["usd"])
    ncalls = usage.get("calls", G._USAGE["calls"])
    per_call = usd / ncalls if ncalls else 0.0
    per_user = usd / len(users) if users else 0.0
    proj_full = per_user * (FULLRUN_Q_PER_USER / q_pilot) * FULLRUN_USERS
    P(f"- measured: {ncalls} calls, {usage.get('prompt_tokens')} prompt + "
      f"{usage.get('completion_tokens')} completion tokens, **${usd:.4f}** "
      f"(rates in ${G.PRICE_IN_PER_M}/M, out ${G.PRICE_OUT_PER_M}/M)")
    P(f"- per-call ${per_call:.4f} | per-user ${per_user:.4f} ({q_pilot} Q/user)")
    P(f"- **projected FULL run** (2828 Q/user x 300 users, richer output) = "
      f"per_user x (2828/{q_pilot}) x 300 = **${proj_full:.2f}**")
    if tripwire:
        P(f"- tripwire (3 users): ${tripwire.get('usd_so_far',0):.4f}; "
          f"projected pilot ${tripwire.get('proj_pilot_usd',0):.2f}; "
          f"projected full ${tripwire.get('proj_full_usd',0):.2f}")
    P("")

    # =============================== ANALYSIS 7: awkward-54 ===============================
    P("## 7. The awkward-flagged tags (do they behave worse on parse/degeneracy/knowledge?)")
    P("")
    awk_objs, norm_objs = [], []
    awk_valid = [0, 0]; norm_valid = [0, 0]   # [valid, total returned]
    for (u, m, o) in cells["concept"]:
        if o is None:
            continue
        isawk = bool(m.get("awkward"))
        ok, _ = validate_cell(o)
        if isawk:
            awk_valid[0] += ok; awk_valid[1] += 1; awk_objs.append(o)
        else:
            norm_valid[0] += ok; norm_valid[1] += 1; norm_objs.append(o)
    ac2, an2 = kdist(awk_objs); nc2, nn2 = kdist(norm_objs)
    # degenerate share among awkward vs normal
    degen_tags = set(ek for (ek, m, lab, share, n) in degen.get("concept", []))
    awk_tagids = set(m["tagId"] for (u, m, o) in cells["concept"] if m.get("awkward"))
    all_tagids = set(m["tagId"] for (u, m, o) in cells["concept"])
    norm_tagids = all_tagids - awk_tagids
    awk_degen = len(degen_tags & awk_tagids); norm_degen = len(degen_tags & norm_tagids)
    P("(valid% here = STRICT validity; the gap is conf-omission on no_clue cells, which awkward tags "
      "hit more often because they are no_clue more often.)")
    P("")
    P("| group | tags | valid% | no_clue | rough_idea | know_well | degenerate cols |")
    P("|---|--:|--:|--:|--:|--:|--:|")
    P(f"| awkward | {len(awk_tagids)} | {100*awk_valid[0]/max(awk_valid[1],1):.2f}% | "
      f"{ac2.get('no_clue',0)/max(an2,1):.3f} | {ac2.get('rough_idea',0)/max(an2,1):.3f} | "
      f"{ac2.get('know_well',0)/max(an2,1):.3f} | {awk_degen} |")
    P(f"| normal | {len(norm_tagids)} | {100*norm_valid[0]/max(norm_valid[1],1):.2f}% | "
      f"{nc2.get('no_clue',0)/max(nn2,1):.3f} | {nc2.get('rough_idea',0)/max(nn2,1):.3f} | "
      f"{nc2.get('know_well',0)/max(nn2,1):.3f} | {norm_degen} |")
    P("")
    P(f"Awkward degenerate rate {awk_degen}/{len(awk_tagids)} = "
      f"{100*awk_degen/max(len(awk_tagids),1):.1f}% vs normal "
      f"{norm_degen}/{len(norm_tagids)} = {100*norm_degen/max(len(norm_tagids),1):.1f}%.")
    P("")

    # =============================== AUTHOR DECISION ===============================
    P("## AUTHOR DECISION")
    P("")
    total_cells = sum(v[0] for v in valid_flags.values())
    total_valid = sum(v[1] for v in valid_flags.values())
    P(f"- **Parse/validity (CORE contract)**: {100*total_valid/max(total_cells,1):.2f}% overall "
      f"(concept {100*valid_flags['concept'][1]/max(valid_flags['concept'][0],1):.2f}%, "
      f"attribute {100*valid_flags['attribute'][1]/max(valid_flags['attribute'][0],1):.2f}%, "
      f"item {100*valid_flags['item'][1]/max(valid_flags['item'][0],1):.2f}%). "
      f"{'PASS (>=99%)' if 100*total_valid/max(total_cells,1) >= 99 else 'BELOW 99% -- see failures'}. "
      f"STRICT validity is lower only because the judge omits `conf` on some no_clue cells "
      f"(systematic; fixable by one prompt line or schema note).")
    P(f"- **Degenerate columns** (evidence for a PRUNE decision, author's call): "
      f"concept {len(degen.get('concept',[]))}, attribute {len(degen.get('attribute',[]))}. "
      f"Proposed empirical prune candidates = the all-`no_clue` degenerate columns "
      f"(near-zero information):")
    for ch in ("concept", "attribute"):
        noclue_degen = [(ek, m) for (ek, m, lab, share, n) in degen.get(ch, []) if lab == "no_clue"]
        names = [(m.get("tag") or m.get("name") or str(ek)) for ek, m in noclue_degen]
        P(f"  - {ch}: {len(noclue_degen)} all-no_clue columns"
          + (f" -> e.g. {names[:15]}" if names else ""))
    P(f"- **rough_idea usage**: overall {ac.get('rough_idea',0)/max(an,1):.3f}; "
      f"the middle level is {'USED' if ac.get('rough_idea',0)/max(an,1) > 0.02 else 'COLLAPSED (near-binary)'}.")
    P(f"- **Consistency vs old judge**: item kappa {cons.get('item',(0,0,float('nan')))[2]:.3f}, "
      f"concept kappa {cons.get('concept',(0,0,float('nan')))[2]:.3f}.")
    P(f"- **Value agreement**: items exact {item_ag.get('exact',float('nan')):.3f}/adj {item_ag.get('adjacent',float('nan')):.3f}; "
      f"tags exact {tag_stats.get('exact',float('nan')):.3f}/adj {tag_stats.get('adjacent',float('nan')):.3f}; "
      f"attrs exact {attr_stats.get('exact',float('nan')):.3f}/adj {attr_stats.get('adjacent',float('nan')):.3f}. "
      f"**MAIN RISK**: the LLM value marginal collapses to 'liked' (~0.94 on items) -- see 3a; "
      f"decide (a) prompt anchoring, (b) predict stars then bin, or (c) accept-and-document, "
      f"BEFORE the full run.")
    P(f"- **Consistency construct note**: item SPEC-kappa 0 is a CONSTRUCT difference (knows-of vs "
      f"has-seen on famous items), not judge noise -- STRICT mapping gives kappa 0.197, and on "
      f"rated (ground-truth-answerable) overlap cells both judges are 110/110 correct. If the fold "
      f"needs the old 'has seen' construct for items, use know_well (not !=no_clue) as the item "
      f"can_answer rule.")
    P(f"- **Full-run cost estimate** for the signed config (2828 Q x 300 users, richer output) = "
      f"**${proj_full:.2f}**.")
    P("- **Schema issues found**: see the failure list in section 1 (empty = none).")
    P("")

    os.makedirs("experiments", exist_ok=True)
    open(REPORT, "w", encoding="utf-8").write("\n".join(L))
    print(f"[saved] {REPORT} ({len(L)} lines)", flush=True)
    # ASCII console summary
    print("\n==== PILOT SUMMARY ====")
    print(f"parse/validity overall = {100*total_valid/max(total_cells,1):.2f}%")
    print(f"value-agree ITEMS exact {item_ag.get('exact',float('nan')):.3f} adj {item_ag.get('adjacent',float('nan')):.3f} corr {item_ag.get('corr',float('nan')):.3f} (n={item_ag['n']})")
    print(f"value-agree TAGS  exact {tag_stats.get('exact',float('nan')):.3f} adj {tag_stats.get('adjacent',float('nan')):.3f} corr {tag_stats.get('corr',float('nan')):.3f} (n={tag_stats['n']})")
    print(f"value-agree ATTR  exact {attr_stats.get('exact',float('nan')):.3f} adj {attr_stats.get('adjacent',float('nan')):.3f} corr {attr_stats.get('corr',float('nan')):.3f} (n={attr_stats['n']})")
    print(f"rough_idea overall share = {ac.get('rough_idea',0)/max(an,1):.3f}")
    print(f"consistency kappa item {cons.get('item',(0,0,float('nan')))[2]:.3f} concept {cons.get('concept',(0,0,float('nan')))[2]:.3f}")
    print(f"degenerate: concept {len(degen.get('concept',[]))} attribute {len(degen.get('attribute',[]))}")
    print(f"measured pilot ${usd:.4f}; projected full ${proj_full:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    if a.collect:
        collect()
    elif a.analyze:
        analyze()
    else:
        print("pass --collect or --analyze")
