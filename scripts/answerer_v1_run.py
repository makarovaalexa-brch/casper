"""ANSWERER v1 FULL RUN (author-approved) — ML-25M, new knowledge+STARS+conf schema.

Applies the THREE approved prompt fixes from the pilot (casper/experiments/ANSWERER_PILOT.md
AUTHOR DECISION) and reuses the pinned judge machinery of scripts/answerer_pilot.py /
scripts/llm_answerability_gate.py (judge gpt-5.4-mini temp 0, answerer split seed 123, user seed 0).

THE THREE FIXES
  F1 VALUES VIA STARS-THEN-BIN: the judge predicts a star rating (0.5-5.0, half-star steps) for every
     non-no_clue cell; we bin deterministically to the locked 4-level scale (>=4.5 loved / 3.5-4 liked /
     2.5-3 meh / <=2 hated). Cache stores BOTH the raw stars and the binned value.
  F2 PERSON-NAME TAG PHRASING: person genome tags ("hitchcock","amy smart",...) re-phrased to
     "movies directed by X" / "movies starring X" / "movies written by X" / etc., detected against the
     IMDb name universe (.cache/instrument2/attr_membership.json) + a small heuristic. Writes the updated
     .cache/instrument2/tag_questions.json in place with a tag_phrasing_version bump.
  F3 CONF ON EVERY CELL: conf required on EVERY cell including no_clue (one prompt line).

FLOW
  python scripts/answerer_v1_run.py --micro   -> apply F2, run the ~$0.50 micro-check on 5 pilot users,
        print the 4 pre-registered gates. If ALL pass, AUTO-PROCEED to the full run; else STOP.
  python scripts/answerer_v1_run.py --full     -> (also auto-invoked) the 300-user x 2828-Q judged grid.
  python scripts/answerer_v1_run.py --analyze  -> write the validation summary into ANSWERER_V1_RUN.md.

Hard cap: micro + full combined <= $45.  Cost check every 25 users; projected-total > $45 => STOP
(cache .cache/instrument2/answerer_v1_grid.json is resumable).
"""
import os, sys, json, re, argparse, collections, difflib, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings("ignore")
import llm_answerability_gate as G          # pinned judge machinery
import answerer_pilot as P                  # reuse inputs/selection/validation helpers

# ------------------------------------------------------------------ pins / paths
SPLIT_SEED = G.ANSWERER_SPLIT_SEED          # 123
USER_SEED = 0
BATCH = 260                                 # ~250-270 Q/call
HARD_CAP_USD = 45.0
FULL_USERS = 300
COST_CHECK_EVERY = 25

CACHE = ".cache/instrument2"
GRID = f"{CACHE}/answerer_v1_grid.json"
USAGE = f"{CACHE}/answerer_v1_usage.json"
MICRO_GRID = f"{CACHE}/answerer_v1_micro_grid.json"
MICRO_USAGE = f"{CACHE}/answerer_v1_micro_usage.json"
TAG_Q = f"{CACHE}/tag_questions.json"
ATTR_MEMB = f"{CACHE}/attr_membership.json"
ATTR_BATTERY = f"{CACHE}/attr_battery.json"
ITEM_LISTS = f"{CACHE}/item_lists.json"
PILOT_GRID = f"{CACHE}/answerer_pilot_grid.json"
GATE_GRID = f"{CACHE}/answerability_grid_ml25m.json"
REPORT = "experiments/ANSWERER_V1_RUN.md"
REPHRASE_LIST = f"{CACHE}/answerer_v1_tag_rephrasings.json"

KNOW = ["no_clue", "rough_idea", "know_well"]
VAL = ["hated", "meh", "liked", "loved"]
VAL_ORD = {v: i for i, v in enumerate(VAL)}
TAG_PHRASING_VERSION = "f2-person-2026-07-08"

# ------------------------------------------------------------------ NEW judge prompt (F1 stars + F3 conf)
SYS_ANSWER_V1 = (
    "You simulate ONE specific movie viewer answering a cold-start preference interview. You are shown a "
    "SAMPLE of this viewer's past star ratings (they have seen many more films than listed). For each "
    "question about a FILM, a PERSON (director/actor/composer/writer), a FILM SERIES, or a KIND of film "
    "(concept), decide how well THIS viewer knows the entity and, if they know it, what STAR RATING they "
    "would give it.\n"
    "Fields per question:\n"
    "  knowledge: one of\n"
    "    'no_clue'    = never heard of it / cannot give an opinion (a refusal). stars OMITTED.\n"
    "    'rough_idea' = knows of it / barely remembers / 'I think I liked it?'. stars present but uncertain.\n"
    "    'know_well'  = has seen it / knows the person's work / confident opinion. stars present.\n"
    "  stars: the rating THIS viewer would give, on a 0.5 to 5.0 scale in HALF-STAR steps "
    "(0.5,1.0,1.5,...,5.0). REQUIRED whenever knowledge is not 'no_clue'; OMIT entirely when knowledge is "
    "'no_clue'. Use the FULL range HONESTLY: a typical viewer rates most films they've seen between 2.0 "
    "and 3.5 stars and reserves 4.5-5.0 for genuine favourites and 0.5-1.5 for films they disliked. Do "
    "NOT default everything to a middling 3.5-4.0 'liked' rating -- spread the ratings the way a real "
    "person's rating history looks.\n"
    "  conf: your confidence in this judgement, 0..1. ALWAYS include conf, on EVERY question, INCLUDING "
    "every 'no_clue'.\n"
    "Judge knowledge HONESTLY and per THIS viewer -- do NOT assume famous = known. Film composers and "
    "screenwriters are usually 'no_clue' for ordinary viewers even when the films are famous; niche "
    "concepts and obscure films are often 'no_clue' too.\n"
    'Output STRICT JSON: {"answers":[{"q":<int index>,"knowledge":"...","stars":<0.5..5.0>,"conf":<0..1>}]} . '
    "One object per question, no prose.")


# ================================================================== F2: person-tag phrasing
def _norm(s):
    s = s.lower().strip()
    s = re.sub(r"[’'`]", "", s)
    s = re.sub(r"[.]", " ", s)            # periods -> spaces (keeps initials structure, e.g. c.s. lewis)
    s = re.sub(r"[-_/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# common words that also happen to be surnames -> never treat a bare single token as a person
_SINGLE_STOP = set((
    "story arnold zombie camp ireland love christian disney german israel gore chase dance bond france "
    "brothers water fire king dick gold silver black white red blue green long short cool good bad best "
    "worst new old dark light epic classic cult indie war god city world life star moon blood death home "
    "school work road wall night day time space future past india china japan egypt mexico canada russia "
    "spain italy poland brazil cuba iran iraq korea").split())

_ROLE_PHRASE = {"director": "directed by", "actor": "starring", "writer": "written by",
                "composer": "with music by", "author": "based on works by"}
_ROLE_BOOST = {"director": 1.3, "actor": 1.2, "writer": 1.0, "composer": 1.0}


def _build_name_universe():
    am = json.load(open(ATTR_MEMB))
    ents = am["entities"]
    full = collections.defaultdict(list)      # norm full name -> [(type, nmov)]
    disp = {}                                  # norm -> (canonical display, nmov)
    dirsur = collections.defaultdict(list)     # surname -> [(canonical, nmov)] (DIRECTORS only)
    allnames = set()
    for typ, lst in ents.items():
        for e in lst:
            nm = _norm(e["name"]); nmov = int(e.get("n_movies", 0))
            full[nm].append((typ, nmov)); allnames.add(nm)
            if nm not in disp or nmov > disp[nm][1]:
                disp[nm] = (e["name"], nmov)
            toks = nm.split()
            if typ == "director" and len(toks) >= 2:
                dirsur[toks[-1]].append((e["name"], nmov))
    return full, disp, dirsur, sorted(allnames)


def _best_type(cands):
    return sorted(cands, key=lambda x: -(x[1] * _ROLE_BOOST.get(x[0], 1.0)))[0][0]


def detect_person(tag, uni):
    """Return (role, display_name) if `tag` is a person/creator tag, else None. Precision-first."""
    full, disp, dirsur, allnames = uni
    author = tag.startswith("author:")
    core = tag.split(":", 1)[1].strip() if author else tag
    n = _norm(core); toks = n.split()
    if author:                                                   # explicit author prefix
        return ("author", core.title())
    # initials + surname pattern, e.g. "c.s. lewis", "j.r.r. tolkien" (ALPHA tokens only)
    if (len(toks) >= 2 and all(len(x) == 1 and x.isalpha() for x in toks[:-1])
            and len(toks[-1]) >= 2 and toks[-1].isalpha()):
        return ("author", core.title())
    # "<Name> brothers" troupe (Coen/Marx/...) -> role-agnostic "by the ..."
    if n.endswith(" brothers"):
        return ("brothers", core.title())
    # full-name exact match (multi-token) -> high precision
    if n in full and len(toks) >= 2:
        typ = _best_type(full[n]); return (typ, disp.get(n, (core, 0))[0])
    # single-token DIRECTOR surname (auteur convention), famous + not a common word
    if len(toks) == 1 and n in dirsur and n not in _SINGLE_STOP and len(n) >= 5:
        cand = sorted(dirsur[n], key=lambda x: -x[1])[0]
        if cand[1] >= 11:
            return ("director", cand[0])
    # fuzzy multi-token (typo tolerance, e.g. "francis ford copolla")
    if len(toks) >= 2:
        m = difflib.get_close_matches(n, allnames, n=1, cutoff=0.9)
        if m:
            typ = _best_type(full[m[0]]); return (typ, disp.get(m[0], (core, 0))[0])
    return None


def person_question(role, name):
    if role == "brothers":
        return f"Do you like movies by the {name}?"
    return f"Do you like movies {_ROLE_PHRASE[role]} {name}?"


def apply_f2():
    """Detect person tags, rewrite their `question` in tag_questions.json IN PLACE (version-bumped).
    Idempotent: re-detects from the tagId->tag mapping every time and rewrites deterministically."""
    uni = _build_name_universe()
    tq = json.load(open(TAG_Q))
    rephrasings = []
    for t in tq["tags"]:
        d = detect_person(t["tag"], uni)
        if not d:
            continue
        role, name = d
        newq = person_question(role, name)
        rephrasings.append(dict(tagId=t["tagId"], tag=t["tag"], role=role, name=name,
                                old_question=t.get("question"), new_question=newq))
        t["question"] = newq
        t["person_role"] = role
        t["person_name"] = name
    tq["tag_phrasing_version"] = TAG_PHRASING_VERSION
    tq["n_person_rephrased"] = len(rephrasings)
    json.dump(tq, open(TAG_Q, "w"), ensure_ascii=False, indent=0)
    json.dump(dict(version=TAG_PHRASING_VERSION, n=len(rephrasings), rephrasings=rephrasings),
              open(REPHRASE_LIST, "w"), ensure_ascii=False, indent=1)
    print(f"[F2] re-phrased {len(rephrasings)} person tags -> {TAG_Q} "
          f"(version {TAG_PHRASING_VERSION}); list -> {REPHRASE_LIST}", flush=True)
    for r in rephrasings:
        print(f"    [{r['role']:8}] {r['tag']!r:42} -> {r['new_question']}", flush=True)
    return rephrasings


# ================================================================== schema validation (v1: stars)
def bin_stars(s):
    """F1 deterministic star -> 4-level value (same table as rating_to_scale)."""
    if s >= 4.5:
        return "loved"
    if s >= 3.5:
        return "liked"
    if s >= 2.5:
        return "meh"
    return "hated"


def parse_cell(o):
    """Normalize a raw judge object into the cache cell {knowledge, stars?, value?, conf}.
    Returns (cell_or_None, core_ok, conf_ok)."""
    if not isinstance(o, dict):
        return None, False, False
    k = o.get("knowledge")
    if k not in KNOW:
        return None, False, False
    # conf (F3)
    conf_ok = False; conf = None
    try:
        conf = float(o.get("conf"))
        conf_ok = 0.0 <= conf <= 1.0
    except (TypeError, ValueError):
        conf_ok = False
    if k == "no_clue":
        cell = {"knowledge": "no_clue"}
        if conf_ok:
            cell["conf"] = conf
        return cell, True, conf_ok
    # stars required
    try:
        stars = float(o.get("stars"))
    except (TypeError, ValueError):
        return None, False, conf_ok
    if not (0.5 <= stars <= 5.0):
        return None, False, conf_ok
    cell = {"knowledge": k, "stars": stars, "value": bin_stars(stars)}
    if conf_ok:
        cell["conf"] = conf
    return cell, True, conf_ok


# ================================================================== per-user judged pass
def run_user_v1(u, split, D, Q, masked):
    """One user: batched LLM calls with the NEW prompt; parse stars->value. Returns ans dict + counts."""
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
        arr = G._call_json(SYS_ANSWER_V1, user_msg)
        for o in arr:
            if isinstance(o, dict) and "q" in o:
                try:
                    li = int(o["q"])
                except (ValueError, TypeError):
                    continue
                if 0 <= li < len(chunk):
                    cell, _, _ = parse_cell(o)
                    ans[start + li] = cell if cell is not None else {"_raw": o}
    return ans, n_shown, N_total


# ================================================================== question assembly (full run)
def build_full_questions(u, split, D, tags, battery, top1000, top1000_set):
    """1128 tags (F2 phrasings) + 700 attrs + 1000 items; mask top1000 items the user rated (known half)."""
    kn, _ = split[u]
    Q = []
    for t in tags:
        Q.append(("concept", dict(tagId=t["tagId"], tag=t["tag"], awkward=bool(t["awkward"]),
                                  prior=t["answer_rate_prior"], memb=t["membership_size"],
                                  person_role=t.get("person_role")),
                  f"[concept] {t['question']}"))
    for e in battery:
        Q.append(("attribute", dict(entity_id=e["entity_id"], atype=e["type"], name=e["name"],
                                    n_movies=e.get("n_movies")),
                  f"[{e['type']}] {e['question']}"))
    masked = [int(j) for j in top1000 if j in kn]                   # rated top1000 items -> masked
    for j in top1000:
        Q.append(("item", dict(j=int(j), cnt=float(D["cnt"][j]), title=D["title"][j]),
                  f"[film] '{D['title'][j]}'"))
    return Q, masked


# ================================================================== usage accounting (sidecar baseline)
_USAGE_BASE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "usd": 0.0}


def _load_usage_base(path):
    global _USAGE_BASE
    if os.path.exists(path):
        b = json.load(open(path))
        _USAGE_BASE = {k: b.get(k, 0) for k in ("calls", "prompt_tokens", "completion_tokens", "usd")}
    else:
        _USAGE_BASE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "usd": 0.0}


def _save_usage(path):
    cum = {k: _USAGE_BASE[k] + G._USAGE[k] for k in ("calls", "prompt_tokens", "completion_tokens", "usd")}
    cum["snapshot"] = G._USAGE["snapshot"]
    json.dump(cum, open(path, "w"), indent=1)


def _cum_usd():
    return _USAGE_BASE["usd"] + G._USAGE["usd"]


# ================================================================== MICRO-CHECK
def pick_micro_users():
    pilot = json.load(open(PILOT_GRID))["users"]
    return sorted(int(u) for u in pilot)[:5]


def the_29_noclue_columns():
    """Recompute the all-no_clue degenerate concept columns from the pilot grid (deterministic)."""
    grid = json.load(open(PILOT_GRID))["users"]
    col = collections.defaultdict(dict)
    for us, rec in grid.items():
        ans = {int(k): v for k, v in rec["ans"].items()}
        for i, (k, m) in enumerate(rec["Q"]):
            if k != "concept":
                continue
            o = ans.get(i)
            if o is None or o.get("knowledge") is None:
                continue
            col[m["tag"]][int(us)] = o["knowledge"]
    out = []
    for tag, um in col.items():
        labs = list(um.values()); n = len(labs)
        if n < 10:
            continue
        cc = collections.Counter(labs); top, tn = cc.most_common(1)[0]
        if top == "no_clue" and tn / n >= 0.95:
            out.append(tag)
    return sorted(out)


def micro_check():
    print("=" * 78, flush=True)
    print("MICRO-CHECK (5 pilot users; re-judge 100 pilot items + 29 no_clue cols + 20 control tags)",
          flush=True)
    print("=" * 78, flush=True)
    D = G.load_data()
    split = G.build_split(D)
    tags = json.load(open(TAG_Q))["tags"]
    tag_by_name = {t["tag"]: t for t in tags}
    pilot = json.load(open(PILOT_GRID))["users"]
    users = pick_micro_users()
    the29 = the_29_noclue_columns()
    # 20 control tags: normal (non-awkward), not person-rephrased, not in the 29; deterministic rng
    person = {t["tag"] for t in tags if t.get("person_role")}
    pool = [t["tag"] for t in tags if not t["awkward"] and t["tag"] not in person and t["tag"] not in the29]
    rng = np.random.default_rng(123)
    control = sorted(rng.choice(sorted(pool), size=20, replace=False).tolist())
    print(f"[micro] users={users}\n[micro] 29 no_clue cols recomputed (n={len(the29)})\n"
          f"[micro] 20 control tags={control}", flush=True)

    _load_usage_base(MICRO_USAGE)
    grid = {}
    if os.path.exists(MICRO_GRID):
        b = json.load(open(MICRO_GRID))
        if b.get("phrasing_version") == TAG_PHRASING_VERSION:
            grid = b.get("users", {})
            print(f"[micro] resuming {len(grid)} cached users", flush=True)

    tagcols = the29 + control
    for u in users:
        if str(u) in grid:
            continue
        # (a) the 100 pilot items with the SAME masked set (identical protocol)
        prec = pilot[str(u)]
        item_qs, masked = [], set(prec.get("masked", []))
        for k, m in prec["Q"]:
            if k == "item":
                j = int(m["j"])
                item_qs.append(("item", dict(j=j, cnt=float(D["cnt"][j]), title=D["title"][j]),
                                f"[film] '{D['title'][j]}'"))
        # (b) tag columns (29 + 20) with CURRENT (F2) phrasing
        tag_qs = []
        for tg in tagcols:
            t = tag_by_name[tg]
            tag_qs.append(("concept", dict(tagId=t["tagId"], tag=tg, person_role=t.get("person_role")),
                           f"[concept] {t['question']}"))
        Q = item_qs + tag_qs
        ans, n_shown, N_total = run_user_v1(u, split, D, Q, sorted(masked))
        grid[str(u)] = dict(masked=sorted(int(j) for j in masked),
                            Q=[[k, {kk: (int(vv) if isinstance(vv, np.integer) else vv)
                                    for kk, vv in m.items()}] for k, m, _ in Q],
                            ans={str(i): o for i, o in ans.items()})
        json.dump(dict(phrasing_version=TAG_PHRASING_VERSION, users=grid), open(MICRO_GRID, "w"))
        _save_usage(MICRO_USAGE)
        print(f"  [micro] user {u}: {len(Q)} Q ({len(item_qs)} items /{len(tag_qs)} tags), "
              f"parsed {len(ans)}, masked {len(masked)} | cum ${_cum_usd():.4f}", flush=True)

    return compute_micro_gates(D, grid, users, the29, control)


def compute_micro_gates(D, grid, users, the29, control):
    # G1 parse/core-contract + G4 conf presence (over ALL micro cells)
    n_cells = n_core = n_conf = 0
    # G2 masked items: (stars, real_rating), liked share
    stars_real = []; liked = 0; nval_item = 0
    # G3 person cols: tag -> set of users with non-no_clue
    col_nonnoclue = collections.defaultdict(set)
    for u in users:
        rec = grid[str(u)]
        rat = dict(D["rat_by_u"][u]); masked = set(rec["masked"])
        ans = {int(k): v for k, v in rec["ans"].items()}
        for i, (k, m) in enumerate(rec["Q"]):
            o = ans.get(i)
            n_cells += 1
            if o is None or "_raw" in o:
                continue
            cell, core_ok, _ = (o, ("knowledge" in o and o["knowledge"] in KNOW), None)
            # re-derive core/conf robustly from stored cell
            kk = o.get("knowledge")
            core = kk in KNOW and (
                (kk == "no_clue" and "stars" not in o) or (kk != "no_clue" and o.get("value") in VAL))
            conf = ("conf" in o)
            n_core += 1 if core else 0
            n_conf += 1 if conf else 0
            if k == "item" and kk != "no_clue" and m["j"] in masked and m["j"] in rat and "stars" in o:
                stars_real.append((float(o["stars"]), float(rat[m["j"]])))
                nval_item += 1
                if o.get("value") == "liked":
                    liked += 1
            if k == "concept" and kk in KNOW and kk != "no_clue":
                col_nonnoclue[m["tag"]].add(u)

    g1 = 100.0 * n_core / max(n_cells, 1)
    g4 = 100.0 * n_conf / max(n_cells, 1)
    if len(stars_real) >= 3:
        a = np.array([x for x, _ in stars_real]); b = np.array([y for _, y in stars_real])
        g2_corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    else:
        g2_corr = float("nan")
    g2_liked = liked / max(nval_item, 1)
    g3_flipped = sum(1 for tg in the29 if col_nonnoclue.get(tg))
    gates = dict(
        G1=dict(name="parse/core-contract", value=g1, thresh=">=99%", pass_=g1 >= 99.0,
                detail=f"{n_core}/{n_cells} cells"),
        G2=dict(name="value de-collapse", corr=g2_corr, liked_share=g2_liked,
                thresh="corr>=0.35 AND liked<=0.65",
                pass_=(not np.isnan(g2_corr)) and g2_corr >= 0.35 and g2_liked <= 0.65,
                detail=f"n_masked_item={nval_item}"),
        G3=dict(name="person-tags de-degenerate", value=g3_flipped, thresh=">=15 of 29",
                pass_=g3_flipped >= 15, detail=f"{g3_flipped}/29 cols now non-no_clue for >=1/5 users"),
        G4=dict(name="conf present", value=g4, thresh=">=99%", pass_=g4 >= 99.0,
                detail=f"{n_conf}/{n_cells} cells"),
        flipped_cols=sorted(tg for tg in the29 if col_nonnoclue.get(tg)),
        not_flipped=sorted(tg for tg in the29 if not col_nonnoclue.get(tg)),
        n_cells=n_cells, cost_usd=_cum_usd(), users=users, the29=the29, control=control)
    return gates


def print_gate_table(gates):
    print("\n" + "-" * 78, flush=True)
    print("PRE-REGISTERED GATES (ALL must pass to proceed to the full run):", flush=True)
    print("-" * 78, flush=True)
    g = gates
    print(f"  G1 parse/core-contract >= 99%          : {g['G1']['value']:.2f}%  "
          f"[{g['G1']['detail']}]  -> {'PASS' if g['G1']['pass_'] else 'FAIL'}", flush=True)
    print(f"  G2 value de-collapse (corr>=.35 & lik<=.65): corr={g['G2']['corr']:.3f} "
          f"liked={g['G2']['liked_share']:.3f} [{g['G2']['detail']}] -> "
          f"{'PASS' if g['G2']['pass_'] else 'FAIL'}", flush=True)
    print(f"  G3 person-tags de-degenerate (>=15/29)  : {g['G3']['value']}/29  "
          f"-> {'PASS' if g['G3']['pass_'] else 'FAIL'}", flush=True)
    print(f"  G4 conf present >= 99%                  : {g['G4']['value']:.2f}%  "
          f"[{g['G4']['detail']}]  -> {'PASS' if g['G4']['pass_'] else 'FAIL'}", flush=True)
    allpass = all(g[k]["pass_"] for k in ("G1", "G2", "G3", "G4"))
    print("-" * 78, flush=True)
    print(f"  ALL GATES: {'PASS -> proceed to FULL run' if allpass else 'FAIL -> STOP (no full run)'}",
          flush=True)
    print(f"  micro cost so far: ${gates['cost_usd']:.4f}", flush=True)
    print("-" * 78 + "\n", flush=True)
    return allpass


# ================================================================== FULL RUN
def full_run():
    print("=" * 78, flush=True)
    print(f"FULL RUN: {FULL_USERS} users x (1128 tags + 700 attrs + 1000 items)", flush=True)
    print("=" * 78, flush=True)
    D = G.load_data()
    split = G.build_split(D)
    tags = json.load(open(TAG_Q))["tags"]
    battery = json.load(open(ATTR_BATTERY))["entities"]
    top1000 = json.load(open(ITEM_LISTS))["lists"]["top1000"]["ids"]
    top1000_set = set(top1000)
    cohort = sorted(int(u) for u in json.load(open(GATE_GRID))["users"] if int(u) in split)
    print(f"[full] cohort={len(cohort)} users | {len(tags)}+{len(battery)}+{len(top1000)}="
          f"{len(tags)+len(battery)+len(top1000)} Q/user | batch {BATCH}", flush=True)

    _load_usage_base(USAGE)
    grid = {}
    if os.path.exists(GRID):
        b = json.load(open(GRID))
        if b.get("split_seed") == SPLIT_SEED and b.get("model") == G.MODEL \
                and b.get("phrasing_version") == TAG_PHRASING_VERSION:
            grid = b.get("users", {})
            print(f"[full] resuming {len(grid)} cached users (prior cum ${_USAGE_BASE['usd']:.4f})",
                  flush=True)

    def flush():
        json.dump(dict(split_seed=SPLIT_SEED, model=G.MODEL, snapshot=G._USAGE["snapshot"],
                       phrasing_version=TAG_PHRASING_VERSION, n_tags=len(tags), n_attrs=len(battery),
                       n_items=len(top1000), users=grid), open(GRID, "w"))

    done = len(grid)
    processed_this_run = 0
    for n, u in enumerate(cohort):
        if str(u) in grid:
            continue
        Q, masked = build_full_questions(u, split, D, tags, battery, top1000, top1000_set)
        ans, n_shown, N_total = run_user_v1(u, split, D, Q, masked)
        grid[str(u)] = dict(n_shown=n_shown, N_total=N_total, masked=[int(j) for j in masked],
                            Q=[[k, {kk: (int(vv) if isinstance(vv, np.integer) else vv)
                                    for kk, vv in m.items()}] for k, m, _ in Q],
                            ans={str(i): o for i, o in ans.items()})
        done += 1; processed_this_run += 1
        flush(); _save_usage(USAGE)
        print(f"  [{done}/{len(cohort)}] user {u}: {len(Q)} Q parsed {len(ans)} masked {len(masked)} "
              f"| cum ${_cum_usd():.4f}", flush=True)
        # ---- cost check every 25 processed users ----
        if processed_this_run % COST_CHECK_EVERY == 0:
            per_user = G._USAGE["usd"] / processed_this_run
            remaining = len(cohort) - done
            proj_total = _cum_usd() + per_user * remaining
            print(f"  [COST] processed {processed_this_run} this run (${per_user:.4f}/user); "
                  f"cum ${_cum_usd():.4f}; projected TOTAL ${proj_total:.2f}", flush=True)
            if proj_total > HARD_CAP_USD:
                print(f"  [COST] STOP: projected total ${proj_total:.2f} > cap ${HARD_CAP_USD}. "
                      f"Cache resumable ({done} users done). Re-run --full to continue.", flush=True)
                return dict(stopped=True, done=done, total=len(cohort), cost=_cum_usd(),
                            proj=proj_total)
    print(f"[full] COMPLETE {done}/{len(cohort)} users; cum ${_cum_usd():.4f}", flush=True)
    return dict(stopped=False, done=done, total=len(cohort), cost=_cum_usd())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--applyf2", action="store_true", help="apply F2 phrasing only")
    ap.add_argument("--micro", action="store_true", help="apply F2 + micro-check + auto-full if gates pass")
    ap.add_argument("--full", action="store_true", help="full run (assumes F2 already applied + gates passed)")
    ap.add_argument("--analyze", action="store_true", help="write ANSWERER_V1_RUN.md validation summary")
    a = ap.parse_args()
    if a.applyf2:
        apply_f2()
    elif a.micro:
        apply_f2()
        gates = micro_check()
        json.dump(gates, open(f"{CACHE}/answerer_v1_micro_gates.json", "w"), indent=1, default=str)
        proceed = print_gate_table(gates)
        if proceed:
            print(">>> ALL GATES PASSED -> AUTO-PROCEEDING TO FULL RUN\n", flush=True)
            full_run()
        else:
            print(">>> GATES FAILED -> STOPPING. No full run.", flush=True)
    elif a.full:
        full_run()
    elif a.analyze:
        print("run analyze() -- see answerer_v1_run.analyze (built after full run)")
    else:
        print("pass --micro (recommended), --applyf2, --full, or --analyze")
