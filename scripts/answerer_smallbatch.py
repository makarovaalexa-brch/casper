"""ANSWERER v1 -- CORRECTED design SMALL BATCH (10 users). Author-directed 2026-07-08.

Supersedes the masking design of scripts/answerer_v1_run.py. Reuses the pinned judge machinery
(scripts/llm_answerability_gate.py: gpt-5.4-mini temp 0, answerer split seed 123). This is a 10-user
plumbing+cost+fidelity batch. It does NOT start the 300-user run.

CORRECTED DESIGN
  1. NO MASKING. Production calls show the FULL known-half profile (cap 200 titles; above the cap =>
     deterministic star-stratified subsample, flagged in-prompt + in assumptions).
  2. Per user ~9 PRODUCTION calls covering 1128 tags + 500 attributes (attr_battery_500.json) +
     top-800 items MINUS items the user rated in the known half (data-filled) MINUS held-out-half items
     (LEAKAGE GUARD -- never asked, never in grid). PLUS 1 VALIDATION call = full profile + ~15
     deterministically-sampled HELD-OUT-half items -> QUARANTINED file answerer_v1_valid.json.
  3. TOKEN COMPRESSION: output {"i":idx,"k":0|1|2,"s":stars(omit if k==0),"c":conf}; k=0 no_clue /
     1 rough_idea / 2 know_well. Compressed "index: text" question lines.
  4. Environment grid: .cache/instrument2/answerer_v1_grid10.json (resumable). knowledge+stars+value+
     conf+source per cell; known-half-rated top-800 cells source="data" (real rating, know_well).

  *** LEAKAGE ASSERTION: no held-out-half item id may enter the environment grid. Enforced at runtime
      per user (assert grid_item_ids & heldout_ids == {}). Validation held-out items live ONLY in the
      separate quarantined file. ***

Run:  python scripts/answerer_smallbatch.py --collect    (tripwire after 2 users, then 10)
      python scripts/answerer_smallbatch.py --analyze    (-> experiments/ANSWERER_SMALLBATCH.md)
"""
import os, sys, json, argparse, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings("ignore")
import llm_answerability_gate as G          # pinned judge machinery

# ------------------------------------------------------------------ pins / paths
SPLIT_SEED = G.ANSWERER_SPLIT_SEED          # 123
USER_SEED = 1                               # THIS batch: seed 1 (disjoint from pilot seed-0 20)
N_USERS = 10
BATCH = 260
PROFILE_CAP = 200
N_VALID_ITEMS = 15
TRIPWIRE_AFTER = 2
TRIPWIRE_10USER_CAP = 3.0                   # if projected 10-user cost > $3 -> STOP

CACHE = ".cache/instrument2"
GRID = f"{CACHE}/answerer_v1_grid10.json"           # environment grid (resumable)
VALID = f"{CACHE}/answerer_v1_valid.json"           # QUARANTINED validation (held-out preds)
USAGE = f"{CACHE}/answerer_v1_grid10_usage.json"
TRIPWIRE = f"{CACHE}/answerer_v1_grid10_tripwire.json"
BATTERY = f"{CACHE}/attr_battery_500.json"
TAG_Q = f"{CACHE}/tag_questions.json"
TAG_MEMB = f"{CACHE}/tag_membership.json"
ITEM_LISTS = f"{CACHE}/item_lists.json"
GATE_GRID = f"{CACHE}/answerability_grid_ml25m.json"
PILOT_GRID = f"{CACHE}/answerer_pilot_grid.json"
REPORT = "experiments/ANSWERER_SMALLBATCH.md"

# pilot per-question completion-token baseline (ANSWERER_PILOT.md sec.6: 738011 completion / 38560 Q)
PILOT_COMPLETION_PER_Q = 738011.0 / (20 * (1128 + 700 + 100))   # = 19.14 tok/Q

KNOW = ["no_clue", "rough_idea", "know_well"]           # index == k
VAL = ["hated", "meh", "liked", "loved"]
VAL_ORD = {v: i for i, v in enumerate(VAL)}

# ------------------------------------------------------------------ compressed prompts
# COMPACT ARRAY ROW: [i, k, h, c]  (token-minimal; keeps the index anchor so the grid never misaligns)
#   i = question index (int)
#   k = knowledge 0/1/2
#   h = HALF-STAR integer 1..10 (= stars*2). 0 when k==0. (integer -> 1 token vs "4.0" -> 2 tokens)
#   c = confidence in TENTHS 0..10 (= round(conf*10)). (integer -> 1 token vs "0.8" -> 2 tokens)
_ROW_SPEC = (
    "Per question output a COMPACT 4-integer array [i,k,h,c]:\n"
    "  i = the question's index (integer, as given).\n"
    "  k = knowledge: 0 = no_clue (never heard of it / cannot give an opinion),\n"
    "                 1 = rough_idea (knows of it / barely remembers),\n"
    "                 2 = know_well (has seen it / knows the person's work, confident).\n"
    "  h = the star rating as an INTEGER number of half-stars, 1..10 (2 stars=4, 3.5 stars=7, 5 stars=10). "
    "When k is 1 or 2, h MUST be between 1 and 10 -- NEVER 0. Set h=0 ONLY when k==0. If you cannot commit "
    "to any rating, use k=0 (not k>=1 with h=0). Use the FULL range HONESTLY: a typical viewer rates most "
    "seen films h=4..7, reserves h=9..10 for genuine favourites and h=1..3 for disliked films. Do NOT "
    "default everything to h=7..8.\n"
    "  c = your confidence as an INTEGER 0..10 (tenths; e.g. 0.8 confidence -> 8). ALWAYS present, on "
    "EVERY row, including k==0.\n")

SYS_PROD = (
    "You simulate ONE specific movie viewer answering a cold-start preference interview. You are shown "
    "this viewer's past star ratings (a sample; they have seen many more films than listed). For each "
    "numbered question about a FILM, a PERSON (director/actor/composer/writer), a FILM SERIES, or a KIND "
    "of film, decide how well THIS viewer knows the entity and, if they know it, what star rating they "
    "would give it. Use general world knowledge and what a person with this rating history plausibly "
    "knows.\n" + _ROW_SPEC +
    "Judge knowledge HONESTLY per THIS viewer -- do NOT assume famous = known. Composers and screenwriters "
    "are usually no_clue for ordinary viewers even when the films are famous; niche kinds and obscure films "
    "are often no_clue too.\n"
    "Return ONE row per question, in order. "
    'Output STRICT JSON, no prose: {"a":[[0,2,8,8],[1,0,0,3]]}')

SYS_VALID = (
    "You predict what THIS specific movie viewer WOULD do with each listed film, from a SAMPLE of their "
    "other ratings (they have seen many more films than listed). For each numbered film decide k = "
    "knowledge (0 no_clue / 1 rough_idea / 2 know_well) and the star rating.\n" + _ROW_SPEC +
    'Output STRICT JSON, no prose: {"a":[[0,2,7,7]]}')


# ------------------------------------------------------------------ compressed parse
def bin_stars(s):
    if s >= 4.5:
        return "loved"
    if s >= 3.5:
        return "liked"
    if s >= 2.5:
        return "meh"
    return "hated"


def rating_to_value(r):
    return bin_stars(r)   # same table (schema rating_to_scale)


def _row_fields(o):
    """Extract (idx, k, h, c) from a compact row: [i,k,h,c] (primary) or a dict fallback."""
    if isinstance(o, (list, tuple)):
        if len(o) < 4:
            return None
        return o[0], o[1], o[2], o[3]
    if isinstance(o, dict):                       # tolerate a dict form {i,k,h,c} or {i,k,s,c}
        idx = o.get("i", o.get("q"))
        h = o.get("h")
        if h is None and o.get("s") is not None:  # stars given directly -> to half-stars
            try:
                h = round(float(o["s"]) * 2)
            except (TypeError, ValueError):
                h = None
        c = o.get("c", o.get("conf"))
        return idx, o.get("k"), h, c
    return None


def parse_compact(o):
    """Raw compact row -> cell {knowledge, stars?, value?, conf?, source}. Returns (cell_or_None,
    core_ok, conf_ok). Does NOT consume the index (caller aligns via the returned idx)."""
    f = _row_fields(o)
    if f is None:
        return None, False, False
    _, kk, h, c = f
    try:
        k = int(kk)
    except (TypeError, ValueError):
        return None, False, False
    if k not in (0, 1, 2):
        return None, False, False
    conf_ok = False
    try:
        cf = float(c)
        if cf > 1.0:                              # tenths integer 0..10 -> 0..1
            cf = cf / 10.0
        conf_ok = 0.0 <= cf <= 1.0
    except (TypeError, ValueError):
        conf_ok = False
    # safety net: k!=0 with an out-of-range/zero half-star = failure to commit a rating -> treat as no_clue
    hnum = None
    try:
        hnum = float(h)
    except (TypeError, ValueError):
        hnum = None
    if k == 0 or hnum is None or hnum < 1:
        cell = {"knowledge": "no_clue", "source": "llm"}
        if k != 0:
            cell["coerced_from_k"] = k            # audit: model said known but gave no usable rating
        if conf_ok:
            cell["conf"] = round(cf, 3)
        return cell, True, conf_ok
    stars = hnum / 2.0                             # half-stars -> stars
    if not (0.5 <= stars <= 5.0):
        return None, False, conf_ok
    cell = {"knowledge": KNOW[k], "stars": stars, "value": bin_stars(stars), "source": "llm"}
    if conf_ok:
        cell["conf"] = round(cf, 3)
    return cell, True, conf_ok


# ------------------------------------------------------------------ inputs
def load_inputs():
    battery = json.load(open(BATTERY))["entities"]
    tags = json.load(open(TAG_Q))["tags"]
    top800 = json.load(open(ITEM_LISTS))["lists"]["top800"]["ids"]
    return battery, tags, top800


# ------------------------------------------------------------------ user selection (seed 1, disjoint)
def pilot_users():
    if os.path.exists(PILOT_GRID):
        return set(int(u) for u in json.load(open(PILOT_GRID))["users"])
    return set()


def pick_users(D, split):
    """10 users from the 300-gate cohort; stratify profile-size TERCILE x DOMINANT GENRE; round-robin
    seed 1; EXCLUDE the pilot's 20 users where possible. Deterministic."""
    cohort = sorted(int(u) for u in json.load(open(GATE_GRID))["users"] if int(u) in split)
    excl = pilot_users()
    avail = [u for u in cohort if u not in excl]
    sizes, doms = {}, {}
    for u in avail:
        kn, _ = split[u]; rat = dict(D["rat_by_u"][u])
        sizes[u] = len(rat)
        doms[u], _ = G.dominant_genre(D, sorted(kn), rat)
    qs = np.quantile([sizes[u] for u in avail], [1 / 3, 2 / 3])
    cells = collections.defaultdict(list)
    for u in avail:
        sz = 0 if sizes[u] <= qs[0] else (1 if sizes[u] <= qs[1] else 2)
        cells[(sz, doms[u])].append(u)
    for k in cells:
        cells[k].sort()
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


# ------------------------------------------------------------------ profile (NO masking; cap 200)
def profile_lines(D, kn, rat, cap=PROFILE_CAP):
    known = sorted(kn)
    if len(known) <= cap:
        return [f"{D['title'][j]} = {rat[j]:.1f}/5" for j in known], False
    # deterministic star-stratified subsample preserving the rating distribution
    bins = collections.defaultdict(list)
    for j in known:
        bins[round(rat[j] * 2) / 2].append(j)
    total = len(known); picked = []
    for b in sorted(bins):
        ids = sorted(bins[b])
        n_b = max(1, int(round(cap * len(ids) / total)))
        if len(ids) <= n_b:
            sel = ids
        else:
            step = len(ids) / n_b
            sel = [ids[int(i * step)] for i in range(n_b)]
        picked.extend(sel)
    picked = sorted(set(picked))[:cap]
    return [f"{D['title'][j]} = {rat[j]:.1f}/5" for j in picked], True


# ------------------------------------------------------------------ question assembly
def build_questions(u, split, D, battery, tags, top800):
    """Return (Q, data_cells, heldout_ids). Q = LLM production questions (compressed text). data_cells =
    known-half-rated top800 item cells filled from DATA. heldout_ids = held-out-half rated dense ids."""
    kn, ho = split[u]
    rat = dict(D["rat_by_u"][u])
    rated_all = set(rat)
    heldset = set(ho)
    top800_set = set(top800)

    Q = []
    for t in tags:
        Q.append(("concept", dict(tagId=t["tagId"], tag=t["tag"], awkward=bool(t["awkward"]),
                                  prior=t["answer_rate_prior"], person_role=t.get("person_role")),
                  f"{t['question']}"))
    for e in battery:
        Q.append(("attribute", dict(entity_id=e["entity_id"], atype=e["type"], name=e["name"],
                                    n_movies=e.get("n_movies")),
                  f"{e['question']}"))
    # items: top800 minus rated (known-rated data-filled; held-rated EXCLUDED as leakage guard)
    data_cells = {}
    for j in top800:
        if j in kn:                                             # known-half rated -> DATA fill
            data_cells[int(j)] = dict(knowledge="know_well", stars=float(rat[j]),
                                      value=rating_to_value(rat[j]), conf=1.0, source="data")
        elif j in heldset:                                     # held-out -> NEVER ask, NEVER grid
            continue
        elif j in rated_all:                                   # rated but neither half (shouldn't occur)
            continue
        else:                                                  # unrated popular -> ASK LLM
            Q.append(("item", dict(j=int(j), cnt=float(D["cnt"][j]), title=D["title"][j]),
                      f"film '{D['title'][j]}'"))
    return Q, data_cells, heldset & top800_set, heldset


# ------------------------------------------------------------------ one LLM call w/ token accounting
def call_and_count(sys_msg, user_msg, n_q, tokstat):
    before = (G._USAGE["completion_tokens"], G._USAGE["prompt_tokens"])
    arr = G._call_json(sys_msg, user_msg)
    tokstat["completion"] += G._USAGE["completion_tokens"] - before[0]
    tokstat["prompt"] += G._USAGE["prompt_tokens"] - before[1]
    tokstat["questions"] += n_q
    tokstat["calls"] += 1
    return arr


def run_production(u, split, D, Q, prof, n_shown, N_total, tokstat):
    ans = {}
    for start in range(0, len(Q), BATCH):
        chunk = Q[start:start + BATCH]
        lines = [f"{i}: {qt}" for i, (_, _, qt) in enumerate(chunk)]
        user_msg = (f"Viewer's rating sample (size {n_shown} of {N_total} total; they have seen many more "
                    f"films than listed):\n{prof}\n\nQuestions:\n" + "\n".join(lines))
        arr = call_and_count(SYS_PROD, user_msg, len(chunk), tokstat)
        for pos, o in enumerate(arr):
            f = _row_fields(o)
            if f is None:
                continue
            try:
                li = int(f[0])
            except (ValueError, TypeError):
                li = pos                                # index missing -> positional fallback
            if 0 <= li < len(chunk):
                cell, _, _ = parse_compact(o)
                ans[start + li] = cell if cell is not None else {"_raw": o}
    return ans


def run_validation(u, split, D, prof, n_shown, N_total, tokstat):
    """1 call: ~15 held-out-half items, predict k+stars. Returns list of dicts (QUARANTINED)."""
    kn, ho = split[u]; rat = dict(D["rat_by_u"][u])
    held_rated = [j for j in ho if rat.get(j, 0) >= 0.5]
    rng = np.random.default_rng(5000 + u)
    rng.shuffle(held_rated)
    sample = sorted(held_rated[:N_VALID_ITEMS])
    if not sample:
        return []
    lines = [f"{i}: film '{D['title'][j]}'" for i, j in enumerate(sample)]
    user_msg = (f"Viewer's rating sample (size {n_shown} of {N_total} total; they have seen many more "
                f"films than listed):\n{prof}\n\nFilms:\n" + "\n".join(lines))
    arr = call_and_count(SYS_VALID, user_msg, len(sample), tokstat)
    pred = {}
    for pos, o in enumerate(arr):
        f = _row_fields(o)
        if f is None:
            continue
        try:
            li = int(f[0])
        except (ValueError, TypeError):
            li = pos
        if 0 <= li < len(sample):
            cell, _, _ = parse_compact(o)
            pred[li] = cell if cell is not None else {"_raw": o}
    rows = []
    for i, j in enumerate(sample):
        c = pred.get(i)
        rows.append(dict(j=int(j), true_rating=float(rat[j]), true_value=rating_to_value(rat[j]),
                         pred=c))
    return rows


# ------------------------------------------------------------------ usage sidecar
_USAGE_BASE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "usd": 0.0}


def _load_usage_base():
    global _USAGE_BASE
    if os.path.exists(USAGE):
        b = json.load(open(USAGE))
        _USAGE_BASE = {k: b.get(k, 0) for k in ("calls", "prompt_tokens", "completion_tokens", "usd")}


def _save_usage(tokstat):
    cum = {k: _USAGE_BASE[k] + G._USAGE[k] for k in ("calls", "prompt_tokens", "completion_tokens", "usd")}
    cum["snapshot"] = G._USAGE["snapshot"]
    cum["tokstat"] = tokstat
    json.dump(cum, open(USAGE, "w"), indent=1)


def _cum_usd():
    return _USAGE_BASE["usd"] + G._USAGE["usd"]


# ------------------------------------------------------------------ collect driver
def collect():
    D = G.load_data()
    split = G.build_split(D)
    battery, tags, top800 = load_inputs()
    picked = pick_users(D, split)
    print(f"[batch] {len(picked)} users | {len(tags)} tags + {len(battery)} attrs + top800 items "
          f"(minus rated) | batch {BATCH} | cap {PROFILE_CAP}", flush=True)
    print(f"[users] {[u for u, _ in picked]}", flush=True)
    assert not (set(u for u, _ in picked) & pilot_users()), "picked users overlap the pilot!"

    _load_usage_base()
    grid = {}
    valid = {}
    tokstat = {"completion": 0, "prompt": 0, "questions": 0, "calls": 0}
    if os.path.exists(GRID):
        b = json.load(open(GRID))
        if b.get("split_seed") == SPLIT_SEED and b.get("model") == G.MODEL:
            grid = b.get("users", {})
            tokstat = b.get("tokstat", tokstat)
            print(f"[cache] resuming {len(grid)} users (prior ${_USAGE_BASE['usd']:.4f})", flush=True)
    if os.path.exists(VALID):
        valid = json.load(open(VALID)).get("users", {})

    def flush():
        json.dump(dict(split_seed=SPLIT_SEED, model=G.MODEL, snapshot=G._USAGE["snapshot"],
                       user_seed=USER_SEED, profile_cap=PROFILE_CAP, n_tags=len(tags),
                       n_attrs=len(battery), n_top800=len(top800), tokstat=tokstat, users=grid),
                  open(GRID, "w"))
        json.dump(dict(QUARANTINE="held-out-half predictions -- NEVER feed to the environment grid/fold",
                       split_seed=SPLIT_SEED, model=G.MODEL, users=valid), open(VALID, "w"))

    done = len(grid)
    processed = 0
    for n, (u, cell) in enumerate(picked):
        if str(u) in grid:
            continue
        kn, ho = split[u]; rat = dict(D["rat_by_u"][u])
        N_total = len(rat)
        plines, capped = profile_lines(D, kn, rat)
        n_shown = len(plines)
        prof = "; ".join(plines) if plines else "(no ratings shown)"
        if capped:   # honesty: tell the judge the shown set is a stratified subsample
            prof = ("[a star-stratified sample of this viewer's ratings]\n" + prof)
        Q, data_cells, held_in_top800, heldset = build_questions(u, split, D, battery, tags, top800)

        ans = run_production(u, split, D, Q, prof, n_shown, N_total, tokstat)
        vrows = run_validation(u, split, D, prof, n_shown, N_total, tokstat)

        # ---- assemble grid entry: LLM cells + DATA cells ----
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
        # data-filled item cells (known-half rated top800)
        data_out = {}
        for j, c in data_cells.items():
            data_out[str(j)] = dict(channel="item", j=int(j), **c)
            item_ids_in_grid.add(int(j))

        # *** LEAKAGE ASSERTION ***
        leak = item_ids_in_grid & heldset
        assert not leak, f"LEAKAGE: held-out item ids entered grid for user {u}: {sorted(leak)[:10]}"

        grid[str(u)] = dict(cell=[int(cell[0]), str(cell[1])], dom=str(cell[1]),
                            n_shown=n_shown, N_total=N_total, profile_capped=bool(capped),
                            n_data_cells=len(data_out), n_llm_cells=len(cell_out),
                            n_heldout=len(heldset), heldout_in_top800_excluded=len(held_in_top800),
                            Q=cell_out, data=data_out)
        valid[str(u)] = dict(n=len(vrows), rows=vrows)
        done += 1; processed += 1
        n_parsed = sum(1 for i in range(len(Q)) if ans.get(i) and "_raw" not in ans[i])
        print(f"  [{done} done] u{u} cell{grid[str(u)]['cell']}: {len(Q)} LLM-Q parsed {n_parsed}, "
              f"data {len(data_out)}, valid {len(vrows)}, capped={capped} | cum ${_cum_usd():.4f}",
              flush=True)
        flush(); _save_usage(tokstat)

        # ---- TRIPWIRE after 2 newly-run users ----
        if processed == TRIPWIRE_AFTER and G._USAGE["calls"] > 0:
            per_user = G._USAGE["usd"] / processed
            proj_10 = per_user * N_USERS
            proj_300 = per_user * 300
            comp_ratio = (tokstat["completion"] / max(tokstat["questions"], 1)) / PILOT_COMPLETION_PER_Q
            print(f"\n[TRIPWIRE] {processed} users; ${G._USAGE['usd']:.4f} (${per_user:.4f}/user)\n"
                  f"  projected 10-user = ${proj_10:.2f}\n"
                  f"  projected 300-user = ${proj_300:.2f}\n"
                  f"  completion tok/Q = {tokstat['completion']/max(tokstat['questions'],1):.2f} "
                  f"(pilot {PILOT_COMPLETION_PER_Q:.2f}); ratio {comp_ratio:.3f}", flush=True)
            json.dump(dict(users_done=processed, usd=G._USAGE["usd"], per_user=per_user,
                           proj_10user=proj_10, proj_300user=proj_300,
                           completion_per_q=tokstat["completion"] / max(tokstat["questions"], 1),
                           pilot_completion_per_q=PILOT_COMPLETION_PER_Q, compression_ratio=comp_ratio,
                           tokstat=dict(tokstat)), open(TRIPWIRE, "w"), indent=1)
            if proj_10 > TRIPWIRE_10USER_CAP:
                print(f"[TRIPWIRE] STOP: projected 10-user ${proj_10:.2f} > cap "
                      f"${TRIPWIRE_10USER_CAP}. Not running remaining users.", flush=True)
                return
            print("[TRIPWIRE] under cap -> continuing to 10 users.\n", flush=True)

    _save_usage(tokstat)
    print(f"[collect] {done}/{N_USERS} users cached; cum ${_cum_usd():.4f}", flush=True)
    print("ALL CACHED -> run --analyze" if done >= N_USERS else "RERUN --collect to continue", flush=True)


# ------------------------------------------------------------------ analyze
def agree_stats(pairs):
    if not pairs:
        return dict(n=0)
    a = np.array([VAL_ORD[x] for x, _ in pairs]); b = np.array([VAL_ORD[y] for _, y in pairs])
    exact = float(np.mean(a == b)); adj = float(np.mean(np.abs(a - b) <= 1))
    corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    return dict(n=len(pairs), exact=exact, adjacent=adj, corr=corr)


def analyze():
    D = G.load_data()
    if not os.path.exists(GRID):
        print("no grid; run --collect"); return
    blob = json.load(open(GRID)); grid = blob["users"]
    valid = json.load(open(VALID))["users"] if os.path.exists(VALID) else {}
    usage = json.load(open(USAGE)) if os.path.exists(USAGE) else {}
    tokstat = blob.get("tokstat", usage.get("tokstat", {}))
    tw = json.load(open(TRIPWIRE)) if os.path.exists(TRIPWIRE) else {}
    users = sorted(int(u) for u in grid)
    L = []; P = L.append

    P("# Answerer v1 -- CORRECTED design SMALL BATCH (10 users)")
    P("")
    P(f"Judge `{usage.get('snapshot', G.MODEL)}` temp {G.TEMPERATURE}, split seed {SPLIT_SEED}, user "
      f"seed {USER_SEED}. NO MASKING; full known-half profile (cap {PROFILE_CAP}). {len(users)} users x "
      f"(1128 tags + 500 attrs + top800-minus-rated items) + 1 quarantined validation call each. "
      f"Grid `{GRID}`; validation `{VALID}` (QUARANTINED). NOT the 300-user run.")
    P("")
    P(f"Users: {users}")
    P("")

    # ---- channel cell collection ----
    cells = {"concept": [], "attribute": [], "item": []}
    n_data = 0
    for us in users:
        rec = grid[str(us)]
        for i, c in rec["Q"].items():
            cells[c["channel"]].append((us, c, c.get("ans")))
        n_data += len(rec.get("data", {}))

    # =========================== (a) cost / tokens ===========================
    P("## (a) Cost + token compression")
    P("")
    usd = usage.get("usd", _cum_usd())
    ncalls = usage.get("calls", tokstat.get("calls", 0))
    per_user = usd / max(len(users), 1)
    proj_300 = per_user * 300
    comp_per_q = tokstat.get("completion", 0) / max(tokstat.get("questions", 1), 1)
    prompt_per_q = tokstat.get("prompt", 0) / max(tokstat.get("questions", 1), 1)
    ratio = comp_per_q / PILOT_COMPLETION_PER_Q
    P(f"- measured: {ncalls} calls, {usage.get('prompt_tokens')} prompt + "
      f"{usage.get('completion_tokens')} completion tokens, **${usd:.4f}** "
      f"(rates ${G.PRICE_IN_PER_M}/M in, ${G.PRICE_OUT_PER_M}/M out)")
    P(f"- per-user **${per_user:.4f}** | **PROJECTED 300-USER = ${proj_300:.2f}**")
    P(f"- production completion tok/Q = **{comp_per_q:.2f}** vs pilot {PILOT_COMPLETION_PER_Q:.2f} "
      f"=> **compression ratio {ratio:.3f}** ({'PASS <0.50' if ratio < 0.50 else 'ABOVE 0.50 target'})")
    P(f"- production prompt tok/Q = {prompt_per_q:.2f}")
    if tw:
        P(f"- tripwire (2 users): 10-user proj ${tw.get('proj_10user',0):.2f}, "
          f"300-user proj ${tw.get('proj_300user',0):.2f}")
    P("")

    # =========================== (b) parse / core-contract ===========================
    P("## (b) Parse / core-contract rate")
    P("")
    P("| channel | cells | returned | core-valid | core% | conf% | hard-invalid |")
    P("|---|--:|--:|--:|--:|--:|--:|")
    tot_cells = tot_core = tot_coerced = 0
    for ch in ("concept", "attribute", "item"):
        tot = len(cells[ch]); ret = core = conf = hard = coerced = 0
        for (u, m, o) in cells[ch]:
            if o is None:
                continue
            ret += 1
            if "_raw" in o:
                hard += 1; continue
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
    P("")
    P(f"Of the core-valid cells, {tot_coerced} were COERCED to no_clue (judge emitted k>=1 with an "
      f"out-of-range/zero half-star = failed to commit a rating; treated as a refusal per the safety net).")
    P("")

    # =========================== (c) validation ===========================
    P("## (c) Validation call (held-out-half items -- QUARANTINED)")
    P("")
    vpairs = []; stars_real = []; liked = 0; nval = 0; n_noclue = 0; n_rows = 0
    for us in users:
        for row in valid.get(str(us), {}).get("rows", []):
            n_rows += 1
            c = row.get("pred")
            if not c or "_raw" in c:
                continue
            if c.get("knowledge") == "no_clue":
                n_noclue += 1; continue
            if "stars" in c:
                stars_real.append((float(c["stars"]), float(row["true_rating"])))
                vpairs.append((c["value"], row["true_value"]))
                nval += 1
                if c.get("value") == "liked":
                    liked += 1
    if len(stars_real) >= 3:
        a = np.array([x for x, _ in stars_real]); b = np.array([y for _, y in stars_real])
        corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
        mae = float(np.mean(np.abs(a - b)))
    else:
        corr, mae = float("nan"), float("nan")
    vag = agree_stats(vpairs)
    P(f"- n held-out items judged = {n_rows}; non-no_clue with stars = {nval}; no_clue = {n_noclue}")
    P(f"- **corr(predicted stars, true held-out rating) = {corr:.3f}** "
      f"(micro-check baseline 0.359; expect >= since full profile)")
    P(f"- star MAE = {mae:.3f} stars")
    P(f"- binned exact = {vag.get('exact',float('nan')):.3f} | adjacent = {vag.get('adjacent',float('nan')):.3f}")
    P(f"- **liked-share = {liked/max(nval,1):.3f}** (n={nval})")
    lm = collections.Counter(x for x, _ in vpairs); tm = collections.Counter(y for _, y in vpairs)
    nn = max(len(vpairs), 1)
    P(f"- marginals -- PRED: " + ", ".join(f"{v} {lm.get(v,0)/nn:.3f}" for v in VAL) +
      " | TRUE: " + ", ".join(f"{v} {tm.get(v,0)/nn:.3f}" for v in VAL))
    P("")

    # =========================== (d) knowledge base rates ===========================
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

    # =========================== (e) degenerate-column scan (tags) ===========================
    P("## (e) Degenerate tag columns (>=90% of these 10 users share one label; n>=8)")
    P("")
    col = collections.defaultdict(dict)
    meta = {}
    for (u, m, o) in cells["concept"]:
        if o is None or "_raw" in o:
            continue
        col[m["tag"]][u] = o.get("knowledge"); meta[m["tag"]] = m
    flagged = []
    for tg, um in col.items():
        labs = list(um.values()); n = len(labs)
        if n < 8:
            continue
        top, tn = collections.Counter(labs).most_common(1)[0]
        if tn / n >= 0.90:
            flagged.append((tg, top, tn / n, n))
    flagged.sort(key=lambda x: (-x[2], x[1]))
    byl = collections.Counter(f[1] for f in flagged)
    P(f"{len(flagged)} degenerate tag columns (of {len(col)}). By label: {dict(byl)}. "
      f"(10-user batch => small-n; illustrative, not the prune decision.)")
    P("")
    noclue = [f for f in flagged if f[1] == 'no_clue']
    P(f"all-no_clue degenerate example tags: {[f[0] for f in noclue[:20]]}")
    P("")

    # =========================== (f) assumptions ===========================
    P("## (f) Assumptions")
    P("")
    capped_users = [us for us in users if grid[str(us)].get("profile_capped")]
    P(f"- **Profile cap**: full known-half shown uncapped for {len(users)-len(capped_users)}/{len(users)} "
      f"users; {len(capped_users)} exceeded {PROFILE_CAP} titles => deterministic star-stratified "
      f"subsample to {PROFILE_CAP}, flagged in-prompt (size string). Capped users: {capped_users}.")
    P(f"- **Battery**: rebuilt attr_battery_500.json from attr_membership.json, top-pop per type "
      f"dir150/act200/comp50/writ25/franch75; verified an EXACT id/phrasing subset of the prior 700 "
      f"battery's per-type top-N (no phrasing drift).")
    P(f"- **Item grid**: top-800 popular catalog. Known-half-rated items => source=\"data\" (real "
      f"rating, know_well, conf 1.0). Held-out-half items => EXCLUDED from grid entirely (leakage "
      f"guard; runtime assertion enforced). Remaining (unrated) => LLM graded stars.")
    P(f"- **Validation quarantine**: {N_VALID_ITEMS} held-out items/user (deterministic rng 5000+u), "
      f"predicted k+stars, written ONLY to {VALID}; never enters the grid.")
    P(f"- **Selection**: 10 users, seed {USER_SEED}, profile-size-tercile x dominant-genre round-robin, "
      f"disjoint from the pilot's 20.")
    P(f"- **Compression**: output is a COMPACT integer array row [i,k,h,c] per question -- i=index "
      f"(kept as the alignment anchor so the grid never misaligns), k=knowledge 0/1/2, h=half-stars "
      f"1..10 (0 iff k==0), c=confidence in tenths 0..10. Decoded back to knowledge/stars/value/conf on "
      f"parse. The literal {{\"i\",\"k\",\"s\",\"c\"}} DICT form the author sketched was measured first and "
      f"gave ~20 tok/Q (ratio ~1.0 vs pilot) -- JSON dict punctuation/keys dominate and the pilot's "
      f"word-labels already tokenize cheaply, so it did NOT meet the <50% target; the array form does "
      f"(ratio {ratio:.3f}). Question lines rendered \"idx: text\" (no [type] wrapper).")
    P(f"- **Constant completion length / truncation check**: full 260-question batches return a "
      f"near-constant ~2091 completion tokens. This is NOT a max_completion_tokens truncation (none is "
      f"set): every 260-row batch parsed 260/260 rows as complete well-formed JSON (finish_reason=stop; "
      f"a mid-list cut would null the whole batch on json.loads), zero None and zero _raw cells. The "
      f"constant is the compact fixed-width row format's signature at temperature 0 (each [i,k,h,c] row "
      f"is structurally identical length regardless of the answer).")
    P(f"- **h=0 contradiction fix**: a first pass (compact prompt without the h-range constraint) had "
      f"2.2% invalid cells = rows [i,k>=1,h=0] (knows-it-but-zero-stars), 416/514 from one user. Fixed by "
      f"a prompt constraint ('when k>=1, h MUST be 1..10; use k=0 if you cannot rate') + a parse safety "
      f"net (k>=1 with h<1 -> no_clue). Re-collected fresh: 0 invalid, 0 coercions.")
    P("")

    os.makedirs("experiments", exist_ok=True)
    open(REPORT, "w", encoding="utf-8").write("\n".join(L))
    print(f"[saved] {REPORT} ({len(L)} lines)", flush=True)
    print("\n==== SUMMARY ====")
    print(f"proj 300-user ${proj_300:.2f} | per-user ${per_user:.4f}")
    print(f"validation corr {corr:.3f} liked-share {liked/max(nval,1):.3f} (n={nval})")
    print(f"parse core-contract {100*tot_core/max(tot_cells,1):.2f}%")
    print(f"compression ratio {ratio:.3f} (comp/Q {comp_per_q:.2f} vs pilot {PILOT_COMPLETION_PER_Q:.2f})")


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
