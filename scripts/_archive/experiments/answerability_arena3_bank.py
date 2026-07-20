"""ARENA-3 MODERATE-TILTED ANSWERABILITY BANK (author-approved LLM judging run, 2026-07-08).

WHY: arena-v2's item-probe answerability was ~97% pmodel-synthesized (firewall breach). The
taste-tracking fuel lives in the MODERATE popularity band (recognizable to SOME users, not
blockbusters-everyone-knows). This pass builds ONE FIXED shared bank (~280 items) tilted moderate
and judges it against ALL 300 study users so arena-3 has 100% EXTERNAL answerability labels (zero
model-synthesized cells) on a shared bank.

DESIGN (all pins inherited from the gate/main-study machinery `llm_answerability_gate as G`):
  - Model gpt-5.4-mini (snapshot recorded), temp 0, answerer split seed 123 (known-half shown only),
    maybe->refuse downstream. SAME judge/prompt template as the gate + main study.
  - BANK (deterministic seed 0, shared across all users):
      ~200 moderate-band items (pop pct 0.75-0.97), proportional-with-floor across all 20 genres,
        spread across decades, franchise + non-franchise;
      ~40 popular anchors (top strata pr>=0.97; the top study-cohort-coverage items of the OLD
        160 probe bank, all of which sit in the top strata) = statics/co-known home turf + continuity;
      ~40 items re-judged from the ALREADY-JUDGED main-study sample (highest main-study grid
        frequency => most paired (user,item) cells) = judge-consistency check vs the cached grid.
  - CALLS: 1 batched call/user over the whole ~280-item bank (main study did 270 Q/call). 300 users.
  - COST TRIPWIRE: `--collect --limit 5` first; extrapolate to 300; if projected>$25 the driver
    ABORTS. Usage logged to a sidecar exactly like the gate/main study.
  - CACHE: incremental resumable write to .cache/instrument2/answerability_arena3_grid.json (never
    overwrites the older grids).

Run probe:    python scripts/answerability_arena3_bank.py --collect --limit 5
Run rest:     python scripts/answerability_arena3_bank.py --collect
Run report:   python scripts/answerability_arena3_bank.py --analyze
"""
import os, sys, json, argparse, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings("ignore")
import llm_answerability_gate as G   # reuse load_data / build_split / prompts / _call_json / usage

ANSWERER_SPLIT_SEED = G.ANSWERER_SPLIT_SEED     # 123
BANK_SEED = 0                                    # deterministic bank selection
N_MODERATE = 200
N_ANCHOR = 40
N_OVERLAP = 40
COST_TRIPWIRE_USD = 25.0                         # hard abort if projected total exceeds this

GRID_CACHE = ".cache/instrument2/answerability_arena3_grid.json"
USAGE_SIDE = ".cache/instrument2/answerability_arena3_usage.json"
BANK_JSON = "experiments/arena3_bank.json"
OUT_MD = "experiments/ARENA3_BANK.md"

GATE_GRID = G.GRID_CACHE                                              # study-user roster (300)
MAINSTUDY_GRID = ".cache/instrument2/answerability_mainstudy_grid.json"  # judge-consistency source


# ----------------------------------------------------------------------------- popularity stratum
def pop_stratum(pr):
    if pr >= 0.97:
        return "top"
    if pr >= 0.90:
        return "upper_mod"
    if pr >= 0.75:
        return "lower_mod"
    return "below_mod"


def item_meta(D, j):
    j = int(j)
    yr = G.YEAR_RE.search(D["title"][j] or "") if hasattr(G, "YEAR_RE") else None
    import re
    m = re.search(r"\((\d{4})\)", D["title"][j] or "")
    year = int(m.group(1)) if m else None
    decade = (year // 10 * 10) if year else None
    fr = is_franchise(D["title"][j])
    return dict(j=j, title=D["title"][j], pr=float(D["pr"][j]), cnt=float(D["cnt"][j]),
                stratum=pop_stratum(float(D["pr"][j])), genre=D["dgen"][j][0],
                genres=D["dgen"][j], decade=decade, franchise=int(fr))


import re as _re
_FR_RE = _re.compile(r"(\b(II|III|IV|VI|VII|VIII|IX|XI|XII)\b|:|\bPart\b|\bChapter\b|\bEpisode\b|"
                     r"\bVol\b|\b[2-9]\b)", _re.I)
_YR_RE = _re.compile(r"\((\d{4})\)")


def is_franchise(title):
    base = _YR_RE.sub("", title or "")
    return 1 if _FR_RE.search(base) else 0


# ----------------------------------------------------------------------------- bank construction
def _stratified_pick(cands, k, D, rng):
    """Pick k items from cands spreading across decades and mixing franchise/non-franchise."""
    if k <= 0 or not cands:
        return []
    metas = [item_meta(D, j) for j in cands]
    by_dec = collections.defaultdict(list)
    for mm in metas:
        by_dec[mm["decade"]].append(mm)
    for dec in by_dec:
        rng.shuffle(by_dec[dec])
        # order within decade to alternate franchise flag
        by_dec[dec].sort(key=lambda m: m["franchise"])
    dec_keys = sorted([d for d in by_dec], key=lambda x: (x is None, x))
    picked, ptr = [], {d: 0 for d in dec_keys}
    fr_toggle = 0
    while len(picked) < k and any(ptr[d] < len(by_dec[d]) for d in dec_keys):
        for d in dec_keys:
            if len(picked) >= k:
                break
            lst = by_dec[d]
            # try to honour franchise alternation
            idx = ptr[d]
            chosen = None
            for scan in range(idx, len(lst)):
                if lst[scan]["franchise"] == fr_toggle:
                    chosen = scan
                    break
            if chosen is None and idx < len(lst):
                chosen = idx
            if chosen is not None:
                picked.append(lst[chosen])
                lst.pop(chosen)
                fr_toggle ^= 1
    return [m["j"] for m in picked]


def build_bank(D, split, study_users, main_grid):
    rng = np.random.default_rng(BANK_SEED)
    pr = D["pr"]

    # ---- moderate band: proportional-with-floor across all present genres ----
    mod_idx = [int(j) for j in np.where((pr >= 0.75) & (pr < 0.97))[0]]
    by_g = collections.defaultdict(list)
    for j in mod_idx:
        by_g[D["dgen"][j][0]].append(j)
    present = [g for g in G.GENRES if by_g.get(g)]
    counts = {g: len(by_g[g]) for g in present}
    tot = sum(counts.values())
    # largest-remainder (Hamilton) proportional allocation, then a floor of 3 to cover all genres
    raw = {g: N_MODERATE * counts[g] / tot for g in present}
    alloc = {g: int(np.floor(raw[g])) for g in present}
    rema = {g: raw[g] - alloc[g] for g in present}
    left = N_MODERATE - sum(alloc.values())
    for g in sorted(rema, key=rema.get, reverse=True)[:max(left, 0)]:
        alloc[g] += 1
    # enforce min floor 3 (or availability) so every present genre is represented
    for g in present:
        alloc[g] = min(counts[g], max(3, alloc[g]))
    # rebalance to exactly N_MODERATE (trim largest over-3; add to genre furthest below its raw target)
    while sum(alloc.values()) > N_MODERATE:
        g = max((g for g in present if alloc[g] > 3), key=lambda g: alloc[g], default=None)
        if g is None:
            break
        alloc[g] -= 1
    while sum(alloc.values()) < N_MODERATE:
        cand = [g for g in present if alloc[g] < counts[g]]
        if not cand:
            break
        g = max(cand, key=lambda g: raw[g] - alloc[g])
        alloc[g] += 1

    moderate = []
    for gi, g in enumerate(present):
        sub_rng = np.random.default_rng(BANK_SEED * 1000 + gi)
        moderate += _stratified_pick(by_g[g], alloc[g], D, sub_rng)
    moderate = list(dict.fromkeys(moderate))

    # ---- anchors: OLD 160-bank top study-cohort-coverage items in the top strata ----
    cover = collections.Counter()
    for u in study_users:
        kn, ho = split[u]
        for j in kn:
            cover[int(j)] += 1
    old160 = [j for j, c in cover.most_common() if c >= 3][:160]
    chosen = set(moderate)
    anchors = []
    for j in old160:                                  # already in descending coverage order
        if j in chosen:
            continue
        if pop_stratum(float(pr[j])) == "top":
            anchors.append(int(j)); chosen.add(int(j))
        if len(anchors) >= N_ANCHOR:
            break

    # ---- overlap: highest main-study grid frequency (max paired cells for consistency) ----
    itfreq = collections.Counter()
    for rec in main_grid.values():
        for k, m in rec["Q"]:
            if k == "item":
                itfreq[int(m["j"])] += 1
    overlap = []
    for j, c in itfreq.most_common():
        if j in chosen:
            continue
        overlap.append(int(j)); chosen.add(int(j))
        if len(overlap) >= N_OVERLAP:
            break

    # ---- assemble bank; role priority moderate/anchor/overlap are DISJOINT by construction ----
    role = {}
    for j in moderate:
        role[j] = "moderate"
    for j in anchors:
        role[j] = "anchor"
    for j in overlap:
        role[j] = "overlap"
    order = moderate + anchors + overlap
    bank = []
    for j in order:
        mm = item_meta(D, j)
        mm["role"] = role[j]
        mm["mainstudy_freq"] = int(itfreq.get(j, 0))
        bank.append(mm)
    return bank


# ----------------------------------------------------------------------------- per-user LLM pass
def run_user(D, split, u, bank):
    kn, ho = split[u]
    rat = dict(D["rat_by_u"][u])
    known_ids = sorted(kn)
    prof_lines = G.profile_text(D, known_ids, rat)
    n_shown, N_total = len(prof_lines), len(rat)
    lines = []
    for i, mm in enumerate(bank):
        lines.append(f"{i}: [item] Has this viewer seen / could they give a real opinion on "
                     f"'{mm['title']}'?")
    prof = "; ".join(prof_lines) if prof_lines else "(no ratings shown)"
    user_msg = (f"Viewer's rating sample (size {n_shown} of {N_total} total; they have seen many more "
                f"films than listed):\n{prof}\n\nQuestions:\n" + "\n".join(lines))
    arr = G._call_json(G.SYS_ANSWER, user_msg)
    ans = {}
    for o in arr:
        if isinstance(o, dict) and "q" in o:
            try:
                ans[int(o["q"])] = o
            except (ValueError, TypeError):
                pass
    return ans


def _flush(grid, bank, snapshot):
    tmp = GRID_CACHE + ".tmp"
    json.dump(dict(split_seed=ANSWERER_SPLIT_SEED, model=G.MODEL, snapshot=snapshot,
                   bank_seed=BANK_SEED, n_bank=len(bank),
                   bank=[{k: v for k, v in mm.items()} for mm in bank],
                   users=grid), open(tmp, "w"))
    os.replace(tmp, GRID_CACHE)


def collect(limit=0, chunk=10):
    D = G.load_data()
    split = G.build_split(D)
    study_users = sorted(int(u) for u in json.load(open(GATE_GRID))["users"])
    main_grid = json.load(open(MAINSTUDY_GRID))["users"]
    bank = build_bank(D, split, study_users, main_grid)
    comp = collections.Counter(mm["role"] for mm in bank)
    strat = collections.Counter(mm["stratum"] for mm in bank)
    print(f"[bank] {len(bank)} items | roles {dict(comp)} | strata {dict(strat)}", flush=True)

    grid, snapshot = {}, None
    if os.path.exists(GRID_CACHE):
        b = json.load(open(GRID_CACHE))
        if b.get("split_seed") == ANSWERER_SPLIT_SEED and b.get("model") == G.MODEL \
                and b.get("n_bank") == len(bank):
            grid = b.get("users", {}); snapshot = b.get("snapshot")
            print(f"[cache] resuming {len(grid)} cached users", flush=True)
        else:
            print("[cache] existing grid config MISMATCH; starting fresh (old grid untouched only if "
                  "path differs) -- ABORT to avoid overwrite", flush=True)
            if b.get("users"):
                raise SystemExit("refusing to overwrite an incompatible arena3 grid at " + GRID_CACHE)

    new = 0
    base_usd = G._USAGE["usd"]
    for uindex, u in enumerate(study_users):
        if str(u) in grid:
            continue
        if limit and new >= limit:
            break
        ans = run_user(D, split, u, bank)
        yes = sum(1 for i in range(len(bank)) if G.is_yes(ans.get(i, {})))
        grid[str(u)] = dict(ans={str(i): o for i, o in ans.items()})
        new += 1
        snapshot = G._USAGE["snapshot"]
        print(f"  [{len(grid)} cached | +{new}] user {u}: {len(bank)} Q, yes {yes}/{len(ans)} parsed "
              f"| running ${G._USAGE['usd']:.3f}", flush=True)
        # ---- COST TRIPWIRE after the first 5 fresh users ----
        if new == 5:
            per_user = (G._USAGE["usd"] - base_usd) / 5.0
            projected = per_user * len(study_users)
            print(f"[tripwire] per-user ${per_user:.4f} -> projected total for {len(study_users)} "
                  f"users ${projected:.2f} (cap ${COST_TRIPWIRE_USD})", flush=True)
            _flush(grid, bank, snapshot)
            if projected > COST_TRIPWIRE_USD:
                _accum_usage(new)
                raise SystemExit(f"ABORT: projected ${projected:.2f} exceeds tripwire "
                                 f"${COST_TRIPWIRE_USD}. Stopped after 5 users; grid flushed.")
        if new % chunk == 0:
            _flush(grid, bank, snapshot)
    _flush(grid, bank, snapshot)
    _accum_usage(new)
    n_cached = sum(1 for u in study_users if str(u) in grid)
    print(f"[collect] {n_cached}/{len(study_users)} cached (+{new} new); running ${G._USAGE['usd']:.3f}",
          flush=True)
    print("ALL CACHED -- run --analyze" if n_cached == len(study_users)
          else "RERUN --collect to continue", flush=True)


def _accum_usage(new):
    if new <= 0:
        return
    cum = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "usd": 0.0}
    if os.path.exists(USAGE_SIDE):
        cum = json.load(open(USAGE_SIDE))
    cum["calls"] += G._USAGE["calls"]; cum["prompt_tokens"] += G._USAGE["prompt_tokens"]
    cum["completion_tokens"] += G._USAGE["completion_tokens"]; cum["usd"] += G._USAGE["usd"]
    cum["snapshot"] = G._USAGE["snapshot"]
    json.dump(cum, open(USAGE_SIDE, "w"), indent=1)


# ----------------------------------------------------------------------------- analysis helpers
def _wilson(k, n, z=1.96):
    if n == 0:
        return (None, None, None)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


def _boot_ci(vals, n_boot=5000, seed=0):
    vals = np.asarray(vals, float)
    if len(vals) == 0:
        return (None, None, None)
    rng = np.random.default_rng(seed)
    bs = np.array([vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(n_boot)])
    return (float(vals.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def analyze():
    D = G.load_data()
    split = G.build_split(D)
    if not os.path.exists(GRID_CACHE):
        print("no arena3 grid yet; run --collect first"); return
    blob = json.load(open(GRID_CACHE))
    bank = blob["bank"]
    users_grid = blob["users"]
    bank_by_idx = {i: mm for i, mm in enumerate(bank)}
    study_users = [int(u) for u in users_grid]
    print(f"[analyze] {len(study_users)} users x {len(bank)} bank items", flush=True)

    # ---------- coverage + parse-failure ----------
    n_cells = 0; n_parsed = 0
    for u_s, rec in users_grid.items():
        ans = rec["ans"]
        n_cells += len(bank)
        n_parsed += len(ans)
    parse_fail = 1.0 - n_parsed / max(n_cells, 1)

    # ---------- base answer rate overall + per stratum + per role ----------
    def yes_of(rec, i):
        return G.is_yes(rec["ans"].get(str(i), {}))
    strat_agg = collections.defaultdict(lambda: [0, 0])   # stratum -> [yes, n]
    role_agg = collections.defaultdict(lambda: [0, 0])
    overall = [0, 0]
    for u_s, rec in users_grid.items():
        for i, mm in bank_by_idx.items():
            if str(i) not in rec["ans"]:
                continue
            y = 1 if yes_of(rec, i) else 0
            overall[0] += y; overall[1] += 1
            strat_agg[mm["stratum"]][0] += y; strat_agg[mm["stratum"]][1] += 1
            role_agg[mm["role"]][0] += y; role_agg[mm["role"]][1] += 1
    base_overall = _wilson(overall[0], overall[1])
    base_strat = {s: _wilson(a[0], a[1]) + (a[1],) for s, a in strat_agg.items()}
    base_role = {r: _wilson(a[0], a[1]) + (a[1],) for r, a in role_agg.items()}

    # ---------- VALIDITY GAP (Hole-2): rated vs genre+stratum-matched never-rated ----------
    # per user: known-half-rated bank items (SHOWN in profile) and held-out-rated (NOT shown) vs
    # never-rated bank items matched on stratum+primary-genre. Report BOTH; held-out is the honest one.
    def validity(kind):
        # kind in {"known", "heldout"}; returns per-stratum gap with bootstrap CI over users
        per_user_gap = collections.defaultdict(list)   # stratum -> list of (rated_yes_rate - never_yes_rate)
        rated_cells = collections.defaultdict(lambda: [0, 0])
        never_cells = collections.defaultdict(lambda: [0, 0])
        for u in study_users:
            rec = users_grid[str(u)]
            kn, ho = split[u]
            knset = set(int(x) for x in kn); hoset = set(int(x) for x in ho)
            rat = dict(D["rat_by_u"][u])
            # classify bank items for this user
            rated_here = []      # (idx, meta)
            never = collections.defaultdict(list)   # (stratum,genre) -> [idx]
            for i, mm in bank_by_idx.items():
                if str(i) not in rec["ans"]:
                    continue
                j = mm["j"]
                is_rated = (j in knset) if kind == "known" else (j in hoset and rat.get(j, 0) >= 1)
                is_never = (j not in knset) and (j not in hoset)
                if is_rated:
                    rated_here.append((i, mm))
                elif is_never:
                    never[(mm["stratum"], mm["genre"])].append(i)
            if not rated_here:
                continue
            rng = np.random.default_rng(7000 + u)
            # per-stratum matched pairs
            su_rated = collections.defaultdict(list)
            su_never = collections.defaultdict(list)
            for i, mm in rated_here:
                pool = never.get((mm["stratum"], mm["genre"]), [])
                if not pool:
                    continue
                ni = int(rng.choice(pool))
                su_rated[mm["stratum"]].append(1 if yes_of(rec, i) else 0)
                su_never[mm["stratum"]].append(1 if yes_of(rec, ni) else 0)
            for s in su_rated:
                if su_rated[s]:
                    rr = np.mean(su_rated[s]); nr = np.mean(su_never[s])
                    per_user_gap[s].append(rr - nr)
                    rated_cells[s][0] += int(np.sum(su_rated[s])); rated_cells[s][1] += len(su_rated[s])
                    never_cells[s][0] += int(np.sum(su_never[s])); never_cells[s][1] += len(su_never[s])
        out = {}
        for s in per_user_gap:
            m, lo, hi = _boot_ci(per_user_gap[s])
            rr = rated_cells[s][0] / max(rated_cells[s][1], 1)
            nr = never_cells[s][0] / max(never_cells[s][1], 1)
            out[s] = dict(gap_pts=(m * 100 if m is not None else None),
                          ci_pts=[lo * 100 if lo is not None else None,
                                  hi * 100 if hi is not None else None],
                          rated_rate=rr, never_rate=nr,
                          n_users=len(per_user_gap[s]), n_pairs=rated_cells[s][1])
        return out
    validity_known = validity("known")
    validity_heldout = validity("heldout")

    # ---------- heterogeneity: per-user moderate-band answer-rate spread + simple ICC ----------
    mod_idx = [i for i, mm in bank_by_idx.items() if mm["stratum"] in ("lower_mod", "upper_mod")]
    user_rates = []
    binary_by_user = []
    for u in study_users:
        rec = users_grid[str(u)]
        ys = [1 if yes_of(rec, i) else 0 for i in mod_idx if str(i) in rec["ans"]]
        if ys:
            user_rates.append(np.mean(ys))
            binary_by_user.append(ys)
    user_rates = np.array(user_rates)
    # one-way random-effects ICC on binary answers (between-user var / total)
    grand = np.mean([v for ys in binary_by_user for v in ys])
    n_i = [len(ys) for ys in binary_by_user]
    k = len(binary_by_user)
    means = np.array([np.mean(ys) for ys in binary_by_user])
    ss_between = sum(n_i[a] * (means[a] - grand) ** 2 for a in range(k))
    ss_within = sum(sum((v - means[a]) ** 2 for v in binary_by_user[a]) for a in range(k))
    N = sum(n_i)
    ms_between = ss_between / max(k - 1, 1)
    ms_within = ss_within / max(N - k, 1)
    n0 = (N - sum(n * n for n in n_i) / N) / max(k - 1, 1)
    icc = (ms_between - ms_within) / (ms_between + (n0 - 1) * ms_within) if (ms_between + (n0 - 1) * ms_within) > 0 else 0.0
    hetero = dict(mod_user_rate_mean=float(user_rates.mean()), mod_user_rate_sd=float(user_rates.std()),
                  mod_user_rate_min=float(user_rates.min()), mod_user_rate_max=float(user_rates.max()),
                  icc_user=float(max(icc, 0.0)), n_users=int(len(user_rates)),
                  note="one-way random-effects ICC on binary can_answer over moderate-band bank items")

    # ---------- judge consistency: overlap items vs cached main-study grid ----------
    main_grid = json.load(open(MAINSTUDY_GRID))["users"]
    # build (user,item)->yes for main study
    main_cell = {}
    for u_s, rec in main_grid.items():
        ans = {int(kk): v for kk, v in rec["ans"].items()}
        for i, (k2, m) in enumerate(rec["Q"]):
            if k2 == "item" and i in ans:
                main_cell[(int(u_s), int(m["j"]))] = 1 if G.is_yes(ans[i]) else 0
    overlap_js = set(mm["j"] for mm in bank if mm["role"] == "overlap")
    a_lab, b_lab = [], []
    for u in study_users:
        rec = users_grid[str(u)]
        for i, mm in bank_by_idx.items():
            if mm["j"] not in overlap_js or str(i) not in rec["ans"]:
                continue
            key = (u, mm["j"])
            if key in main_cell:
                a_lab.append(1 if yes_of(rec, i) else 0)
                b_lab.append(main_cell[key])
    a_lab = np.array(a_lab); b_lab = np.array(b_lab)
    if len(a_lab):
        agree = float(np.mean(a_lab == b_lab))
        pa = agree
        p_yes_a = a_lab.mean(); p_yes_b = b_lab.mean()
        pe = p_yes_a * p_yes_b + (1 - p_yes_a) * (1 - p_yes_b)
        kappa = (pa - pe) / (1 - pe) if (1 - pe) > 0 else float("nan")
    else:
        agree, kappa = None, None
    consistency = dict(n_paired_cells=int(len(a_lab)), n_overlap_items=len(overlap_js),
                       pct_agree=agree, cohen_kappa=(float(kappa) if kappa is not None else None),
                       arena3_yes_rate=(float(a_lab.mean()) if len(a_lab) else None),
                       mainstudy_yes_rate=(float(b_lab.mean()) if len(b_lab) else None))

    # ---------- cost ----------
    usage = json.load(open(USAGE_SIDE)) if os.path.exists(USAGE_SIDE) else {}
    cost = dict(calls=usage.get("calls"), prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                usd_estimate=round(usage.get("usd", 0.0), 4),
                usd_per_call=round(usage.get("usd", 0.0) / max(usage.get("calls", 1), 1), 5),
                snapshot=usage.get("snapshot"), price_in_per_m=G.PRICE_IN_PER_M,
                price_out_per_m=G.PRICE_OUT_PER_M)

    # ---------- bank composition ----------
    comp = dict(
        n_items=len(bank),
        roles={r: sum(1 for mm in bank if mm["role"] == r) for r in ("moderate", "anchor", "overlap")},
        strata={s: sum(1 for mm in bank if mm["stratum"] == s)
                for s in ("top", "upper_mod", "lower_mod", "below_mod")},
        genres=dict(collections.Counter(mm["genre"] for mm in bank)),
        decades=dict(collections.Counter(str(mm["decade"]) for mm in bank)),
        franchise=dict(collections.Counter("franchise" if mm["franchise"] else "standalone" for mm in bank)))

    result = dict(
        config=dict(model=G.MODEL, snapshot=cost["snapshot"], temperature=G.TEMPERATURE,
                    answerer_split_seed=ANSWERER_SPLIT_SEED, bank_seed=BANK_SEED,
                    n_users=len(study_users), n_bank=len(bank), maybe_policy="refuse",
                    dataset="ML-25M", grid_cache=GRID_CACHE),
        coverage=dict(users_judged=len(study_users), cells=n_cells, parsed=n_parsed,
                      parse_fail_rate=parse_fail),
        bank_composition=comp,
        base_rate=dict(overall=base_overall, by_stratum=base_strat, by_role=base_role),
        validity_gap=dict(known_half_rated_vs_never=validity_known,
                          heldout_rated_vs_never=validity_heldout,
                          note="known-half-rated items ARE shown in the profile (partly mechanical yes); "
                               "held-out-rated are NOT shown => the honest user-specific-signal number."),
        heterogeneity=hetero,
        judge_consistency=consistency,
        cost=cost,
        artifact=dict(grid=GRID_CACHE, bank_json=BANK_JSON, report=OUT_MD))

    os.makedirs("experiments", exist_ok=True)
    json.dump(dict(config=result["config"], composition=comp,
                   bank=bank), open(BANK_JSON, "w"), indent=1, default=str)
    json.dump(result, open("experiments/arena3_bank_results.json", "w"), indent=1, default=str)
    write_md(result)
    print(f"[saved] {BANK_JSON}\n[saved] {OUT_MD}\n[saved] experiments/arena3_bank_results.json", flush=True)
    _print_summary(result)


def _fmt_ci(t):
    if t is None or t[0] is None:
        return "n/a"
    return f"{t[0]*100:.1f}% [{t[1]*100:.1f}, {t[2]*100:.1f}]"


def write_md(r):
    c = r["config"]; comp = r["bank_composition"]; cost = r["cost"]
    L = []
    L.append("# ARENA-3 MODERATE-TILTED ANSWERABILITY BANK -- FUEL REPORT\n")
    L.append("Date 2026-07-08. Script `scripts/answerability_arena3_bank.py`. Author-approved LLM "
             "judging run: one FIXED shared item bank judged against all 300 study users so arena-3 "
             "has 100% EXTERNAL answerability labels (zero model-synthesized cells) on the moderate "
             "popularity band where the taste-tracking fuel lives.\n")
    L.append("## Config\n")
    L.append(f"- Judge **{c['model']}** (snapshot **{c['snapshot']}**), temperature "
             f"{c['temperature']}, `maybe`->refuse. Answerer split seed {c['answerer_split_seed']} "
             f"(known-half shown only). Bank seed {c['bank_seed']}. Dataset {c['dataset']}.\n")
    L.append(f"- {c['n_users']} study users x {c['n_bank']}-item shared bank; 1 batched call/user.\n")
    L.append("\n## 1. Coverage\n")
    cov = r["coverage"]
    L.append(f"- Users judged: {cov['users_judged']}; cells (users x items): {cov['cells']}; "
             f"parsed {cov['parsed']}; parse-failure rate **{cov['parse_fail_rate']*100:.2f}%**.\n")
    L.append("\n## Bank composition (deterministic, seed 0)\n")
    L.append(f"- Roles: {comp['roles']}  (target 200 moderate / 40 anchor / 40 overlap)\n")
    L.append(f"- Popularity strata: {comp['strata']}\n")
    L.append(f"- Franchise mix: {comp['franchise']}\n")
    L.append(f"- Primary-genre counts: {comp['genres']}\n")
    L.append(f"- Decade counts: {comp['decades']}\n")
    L.append("\n## 2. Base answer rate (Wilson 95% CI)\n")
    L.append(f"- Overall: {_fmt_ci(r['base_rate']['overall'])}\n")
    L.append("\n| stratum | yes-rate [95% CI] | n cells |")
    L.append("|---|---|---|")
    for s in ("top", "upper_mod", "lower_mod", "below_mod"):
        if s in r["base_rate"]["by_stratum"]:
            t = r["base_rate"]["by_stratum"][s]
            L.append(f"| {s} | {_fmt_ci(t[:3])} | {t[3]} |")
    L.append("\n| role | yes-rate [95% CI] | n cells |")
    L.append("|---|---|---|")
    for role in ("anchor", "moderate", "overlap"):
        if role in r["base_rate"]["by_role"]:
            t = r["base_rate"]["by_role"][role]
            L.append(f"| {role} | {_fmt_ci(t[:3])} | {t[3]} |")
    L.append("\n## 3. THE VALIDITY GAP (Hole-2): rated vs stratum+genre-matched never-rated\n")
    L.append("Held-out-rated (NOT shown in profile) is the honest user-specific-signal number; "
             "known-half-rated is shown in the profile (partly mechanical yes), reported for completeness.\n")
    for label, key in (("Held-out-rated vs never (HONEST)", "heldout_rated_vs_never"),
                       ("Known-half-rated vs never (shown; mechanical)", "known_half_rated_vs_never")):
        L.append(f"\n**{label}**\n")
        L.append("| stratum | gap (pts) [95% CI] | rated yes | never yes | n users | n pairs |")
        L.append("|---|---|---|---|---|---|")
        vd = r["validity_gap"][key]
        for s in ("top", "upper_mod", "lower_mod", "below_mod"):
            if s in vd:
                d = vd[s]
                ci = d["ci_pts"]
                cis = f"[{ci[0]:.1f}, {ci[1]:.1f}]" if ci[0] is not None else "n/a"
                L.append(f"| {s} | {d['gap_pts']:.1f} {cis} | {d['rated_rate']*100:.1f}% | "
                         f"{d['never_rate']*100:.1f}% | {d['n_users']} | {d['n_pairs']} |")
    L.append("\n## 4. Heterogeneity (moderate band)\n")
    h = r["heterogeneity"]
    L.append(f"- Per-user moderate-band answer-rate: mean {h['mod_user_rate_mean']*100:.1f}%, "
             f"sd {h['mod_user_rate_sd']*100:.1f}pt, range [{h['mod_user_rate_min']*100:.1f}, "
             f"{h['mod_user_rate_max']*100:.1f}]% over {h['n_users']} users.\n")
    L.append(f"- ICC (between-user variance share, binary answers): **{h['icc_user']:.3f}**.\n")
    L.append("\n## 5. Judge consistency (re-judged overlap items vs cached main-study grid)\n")
    jc = r["judge_consistency"]
    L.append(f"- {jc['n_paired_cells']} paired (user,item) cells over {jc['n_overlap_items']} overlap "
             f"items. **%agree = {jc['pct_agree']*100:.1f}%**, **Cohen kappa = {jc['cohen_kappa']:.3f}**. "
             f"arena3 yes-rate {jc['arena3_yes_rate']*100:.1f}% vs main-study {jc['mainstudy_yes_rate']*100:.1f}%.\n")
    L.append("\n## 6. Cost (exact from response.usage)\n")
    L.append(f"- {cost['calls']} calls, {cost['prompt_tokens']} prompt + {cost['completion_tokens']} "
             f"completion tokens, **${cost['usd_estimate']}** (${cost['usd_per_call']}/call; documented "
             f"rates in ${cost['price_in_per_m']}/M, out ${cost['price_out_per_m']}/M; tokens exact).\n")
    L.append("\n## Artifacts\n")
    for kk, vv in r["artifact"].items():
        L.append(f"- {kk}: `{vv}`\n")
    open(OUT_MD, "w", encoding="utf-8").write("\n".join(L) + "\n")


def _print_summary(r):
    print("\n================= ARENA-3 BANK SUMMARY =================", flush=True)
    print(f"bank {r['config']['n_bank']} items | roles {r['bank_composition']['roles']} | "
          f"strata {r['bank_composition']['strata']}", flush=True)
    print(f"coverage {r['coverage']['users_judged']} users, parse-fail "
          f"{r['coverage']['parse_fail_rate']*100:.2f}%", flush=True)
    print(f"base yes overall {_fmt_ci(r['base_rate']['overall'])}", flush=True)
    for s, t in r["base_rate"]["by_stratum"].items():
        print(f"  stratum {s}: {_fmt_ci(t[:3])} (n={t[3]})", flush=True)
    print("validity gap HELDOUT-rated vs never (honest):", flush=True)
    for s, d in r["validity_gap"]["heldout_rated_vs_never"].items():
        print(f"  {s}: {d['gap_pts']:.1f}pt CI{[round(x,1) for x in d['ci_pts']]} "
              f"(rated {d['rated_rate']*100:.0f}% vs never {d['never_rate']*100:.0f}%, "
              f"{d['n_pairs']} pairs)", flush=True)
    jc = r["judge_consistency"]
    print(f"judge consistency: {jc['pct_agree']*100:.1f}% agree, kappa {jc['cohen_kappa']:.3f} "
          f"({jc['n_paired_cells']} cells)", flush=True)
    print(f"ICC moderate-band {r['heterogeneity']['icc_user']:.3f}", flush=True)
    print(f"COST ${r['cost']['usd_estimate']} ({r['cost']['calls']} calls)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.collect:
        collect(limit=a.limit)
    elif a.analyze:
        analyze()
    else:
        print("pass --collect [--limit N] or --analyze")
