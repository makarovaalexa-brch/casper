"""LLM answerability GATE script (upgrade of scripts/llm_answerability_pilot.py) — ML-25M.

Implements the frozen pre-registration (answerability_PLAN.md §A/B/C, answerability_study_design.md,
answerability_design_review_2026-07-06.md). This file is the machinery for the fuel gate; the MINI-PILOT
mode (--minipilot) runs ~15 users for plumbing sanity and STOPS. The 300-user gate is a separate,
owner-approved spend.

What this implements from the review (the 6 holes + smaller nits):
  Hole 1  masked-rated-item calibration pass -> LLM rating predictor MAE (separate --mask pass).
  Hole 2  validity gap = held-out-RATED vs popularity+genre-MATCHED never-rated answer-rate, within
          popularity tiers (the KEY new machinery).
  Hole 3  G2 exploitability = stub hook (computed on cached grid; mini-pilot only checks plumbing).
  Hole 4  masked-item MAE feeds the counterfactual-value fidelity sigma (recorded, not applied here).
  Hole 5  one FIXED answerer split, cached + recorded (split-sensitivity = separate later pass).
  Hole 6  held-out TARGET items are EXCLUDED from the interview bank; validity/masked are SEPARATE passes.
  nits    ratings labelled correctly (not "liked"); "sample of size n of N (seen many more)"; json_object
          with an "answers" list key (fixes the pilot's json_object-vs-list mismatch); model snapshot +
          temperature pinned; response.usage logged every call with a running $ estimate.

Answerer split convention: reuses the I2 arena profile-split (scripts/instrument2/ml25m_arena.py) — per
user, shuffle ALL rated items, first half = KNOWN/profile (answerer sees), second half = held-out targets.
Per §5 the answerer runs on ONE FIXED split; we pin ANSWERER_SPLIT_SEED=123 (the I2 arena default) and
cache it. Recommender-eval seeds {1,2,3,7,11} are a downstream concern, not used by the answerer.

"maybe" -> refuse (primary), per prereg. can_answer=="yes" is the only positive.

Run mini-pilot:  python scripts/llm_answerability_gate.py --minipilot --n_users 15
"""
import os, re, json, time, argparse, collections
import numpy as np
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

# ----------------------------------------------------------------------------- config / pins
MODEL = os.environ.get("GATE_MODEL", "gpt-5.4-mini")   # resolved snapshot recorded from resp.model
TEMPERATURE = 0.0                                        # PINNED (prereg reproducibility)
ANSWERER_SPLIT_SEED = 123                                # PINNED canonical answerer split (I2 arena default)
USER_SAMPLE_SEED = 0                                     # PINNED user-selection rng
# gpt-5.4-mini price ($/1M tokens). NOTE: token counts below are EXACT (logged usage); the $ is an
# estimate using these documented rates — adjust to the billed rate if it differs. (5-mini-class guess.)
PRICE_IN_PER_M = float(os.environ.get("GATE_PRICE_IN", "0.25"))
PRICE_OUT_PER_M = float(os.environ.get("GATE_PRICE_OUT", "2.00"))

META = "data/movielens/.cache/ml25m/meta.npz"
MOVIES_CSV = "data/movielens/movies.csv"
CONCEPTS_NPZ = ".cache/instrument2/concepts_ml25m.npz"
SPLIT_CACHE = ".cache/instrument2/answerability_answerer_split.json"
OUT_JSON = "experiments/answerability_gate_minipilot_results.json"

GENRES = ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama",
          "Fantasy", "Film-Noir", "Horror", "IMAX", "Musical", "Mystery", "Romance", "Sci-Fi",
          "Thriller", "War", "Western", "(no genres listed)"]
GIX = {g: i for i, g in enumerate(GENRES)}

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ----------------------------------------------------------------------------- usage / cost accounting
_USAGE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "usd": 0.0, "snapshot": None}


def _log_usage(resp):
    u = resp.usage
    _USAGE["calls"] += 1
    _USAGE["prompt_tokens"] += u.prompt_tokens
    _USAGE["completion_tokens"] += u.completion_tokens
    _USAGE["usd"] += u.prompt_tokens / 1e6 * PRICE_IN_PER_M + u.completion_tokens / 1e6 * PRICE_OUT_PER_M
    if _USAGE["snapshot"] is None:
        _USAGE["snapshot"] = resp.model
    print(f"    [usage] call#{_USAGE['calls']} in={u.prompt_tokens} out={u.completion_tokens} "
          f"| running ${_USAGE['usd']:.4f} (snapshot {resp.model})", flush=True)


def _call_json(sys_msg, user_msg, max_retry=2):
    """One chat call, json_object with an 'answers' list; usage logged; returns parsed list."""
    last = ""
    for attempt in range(max_retry + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL, temperature=TEMPERATURE,
                messages=[{"role": "system", "content": sys_msg},
                          {"role": "user", "content": user_msg}],
                response_format={"type": "json_object"})
            _log_usage(resp)
            txt = resp.choices[0].message.content
            last = txt
            obj = json.loads(txt)
            arr = obj.get("answers", obj if isinstance(obj, list) else None)
            if arr is None:  # tolerate a top-level list under any single key
                for v in obj.values():
                    if isinstance(v, list):
                        arr = v; break
            if isinstance(arr, list):
                return arr
        except Exception as e:
            print(f"    [retry {attempt}] {type(e).__name__}: {str(e)[:160]}", flush=True)
            time.sleep(1.0)
    print(f"    PARSE/CALL FAIL: {last[:200]}", flush=True)
    return []


# ----------------------------------------------------------------------------- data loading
def load_data():
    d = np.load(META)
    keepI = d["keepI"].astype(np.int64)          # dense id -> movieId
    cnt = d["cnt"].astype(np.float64)            # train-like popularity per dense id
    ni, nu = int(d["ni"]), int(d["nu"])
    uu, ii, rr = d["uu"], d["ii"], d["rr"]
    va = d["va"].astype(np.int64); te = d["te"].astype(np.int64)

    # movieId -> (title, [genres])
    mv = {}
    with open(MOVIES_CSV, encoding="utf-8") as f:
        next(f)
        for line in f:
            # movieId,title,genres ; title may contain commas -> parse by first comma + last comma
            i0 = line.find(",")
            i1 = line.rfind(",")
            mid = int(line[:i0]); title = line[i0 + 1:i1].strip().strip('"')
            genres = line[i1 + 1:].strip().split("|")
            mv[mid] = (title, genres)

    # dense-id genre matrix + title lookup
    Gmat = np.zeros((ni, len(GENRES)), np.float32)
    title = [None] * ni; dgen = [None] * ni
    for i in range(ni):
        mid = int(keepI[i]); t, gs = mv.get(mid, (f"movie{mid}", ["(no genres listed)"]))
        title[i] = t; dgen[i] = gs
        for g in gs:
            if g in GIX:
                Gmat[i, GIX[g]] = 1.0

    # per-dense-id popularity tier (design anchor: top-5% famous). blend train-like cnt.
    pr = cnt.argsort().argsort() / (ni - 1)   # percentile rank 0..1
    tier = np.where(pr >= 0.95, "famous", np.where(pr >= 0.60, "moderate", "obscure"))

    # eval-cohort rating dicts (va + te users)
    evalset = np.zeros(nu, bool); evalset[va] = True; evalset[te] = True
    sel = evalset[uu]
    rat_by_u = collections.defaultdict(list)
    for u, i, r in zip(uu[sel].tolist(), ii[sel].tolist(), rr[sel].tolist()):
        rat_by_u[u].append((i, float(r)))

    # concepts
    c = np.load(CONCEPTS_NPZ, allow_pickle=True)
    concepts = dict(names=c["tag_names"], item_tag=c["item_tag"], coverage=c["coverage"].astype(np.int64),
                    mass=c["tag_mass"].astype(np.float64))
    return dict(keepI=keepI, cnt=cnt, ni=ni, nu=nu, va=va, te=te, Gmat=Gmat, title=title,
                dgen=dgen, tier=tier, pr=pr, rat_by_u=rat_by_u, concepts=concepts)


# ----------------------------------------------------------------------------- fixed answerer split (§5)
def build_split(D):
    """ONE fixed answerer split, cached. per-user shuffle ALL rated items rng(123): first half KNOWN,
    second half HELD-OUT targets. Requires >=6 rated items (mirrors ml1m/ml25m arena)."""
    if os.path.exists(SPLIT_CACHE):
        blob = json.load(open(SPLIT_CACHE))
        if blob.get("seed") == ANSWERER_SPLIT_SEED:
            print(f"[split] loaded cached answerer split seed={ANSWERER_SPLIT_SEED} "
                  f"({len(blob['split'])} users)", flush=True)
            return {int(u): (set(v["known"]), v["heldout"]) for u, v in blob["split"].items()}
    rs = np.random.default_rng(ANSWERER_SPLIT_SEED)
    split = {}
    for u, v in D["rat_by_u"].items():
        its = list(dict(v))                       # unique dense ids, insertion order
        if len(its) >= 6:
            il = its[:]; rs.shuffle(il)
            split[u] = (set(il[:len(il) // 2]), il[len(il) // 2:])
    json.dump({"seed": ANSWERER_SPLIT_SEED, "convention": "I2 ml25m_arena per-user half-split; "
               "known=first half of rng(123)-shuffled rated items, heldout=second half; >=6 rated",
               "split": {str(u): {"known": sorted(kn), "heldout": ho} for u, (kn, ho) in split.items()}},
              open(SPLIT_CACHE, "w"))
    print(f"[split] built + cached answerer split seed={ANSWERER_SPLIT_SEED} ({len(split)} users) "
          f"-> {SPLIT_CACHE}", flush=True)
    return split


# ----------------------------------------------------------------------------- user stratified sampling
def dominant_genre(D, known_ids, rat):
    g = np.zeros(len(GENRES))
    for j in known_ids:
        if rat[j] >= 4.0:
            g += D["Gmat"][j]
    if g.sum() == 0:
        return "(none)", np.zeros(len(GENRES))
    return GENRES[int(np.argmax(g))], g / g.sum()


def pick_users(D, split, n_users):
    """Stratify eval-cohort (te) users by profile-size tercile x dominant genre (NOT 1-per-genre)."""
    te = [u for u in D["te"].tolist() if u in split]
    sizes = {u: len(dict(D["rat_by_u"][u])) for u in te}
    qs = np.quantile(list(sizes.values()), [1 / 3, 2 / 3])
    rng = np.random.default_rng(USER_SAMPLE_SEED)
    cells = collections.defaultdict(list)
    for u in te:
        kn, _ = split[u]; rat = dict(D["rat_by_u"][u])
        dg, _ = dominant_genre(D, kn, rat)
        sz = 0 if sizes[u] <= qs[0] else (1 if sizes[u] <= qs[1] else 2)
        cells[(sz, dg)].append(u)
    keys = list(cells.keys()); rng.shuffle(keys)
    picked = []
    ci = 0
    while len(picked) < n_users and keys:
        k = keys[ci % len(keys)]
        if cells[k]:
            u = cells[k].pop(rng.integers(len(cells[k])))
            picked.append((u, k))
        ci += 1
        if ci > 10000:
            break
    return picked


# ----------------------------------------------------------------------------- question bank
def concept_bank(D, n=40):
    """~n genome-tag concepts binned by breadth (coverage tiers). FIXED across users."""
    cov = D["concepts"]["coverage"]; names = D["concepts"]["names"]
    order = np.argsort(-cov)
    idx = np.array_split(order, 3)  # broad / mid / niche by coverage rank
    per = n // 3
    chosen = []
    for tier_name, arr in zip(["broad", "mid", "niche"], idx):
        for ci in arr[:per]:
            chosen.append((int(ci), str(names[ci]), tier_name))
    return chosen  # (concept_dense_tag_idx, name, breadth)


def concept_genre_vec(D, ctag_idx):
    """concept genre vector = item_tag-relevance-weighted mean of member item genre vectors."""
    w = D["concepts"]["item_tag"][:, ctag_idx].astype(np.float64)
    if w.sum() == 0:
        return np.zeros(len(GENRES))
    v = (D["Gmat"] * w[:, None]).sum(0) / w.sum()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def designed_item_bank(D, m=30):
    """global designed item bank: popularity strata x genre coverage. FIXED across users."""
    rng = np.random.default_rng(7)
    out = []
    for tier in ["famous", "moderate", "obscure"]:
        pool = np.where(D["tier"] == tier)[0]
        rng.shuffle(pool)
        # spread across primary genres
        seen_g = collections.Counter()
        for j in pool:
            pg = D["dgen"][j][0]
            if seen_g[pg] < max(1, m // 30):
                out.append((int(j), tier)); seen_g[pg] += 1
            if len([o for o in out if o[1] == tier]) >= m // 3:
                break
    return out


def taste_adjacent_items(D, known_ids, rat, dg_vec, k=8):
    """per-user taste-adjacent: popular unrated items in the user's dominant genres."""
    rated = set(known_ids)
    scores = D["Gmat"] @ dg_vec + 0.15 * D["pr"]   # genre-match + mild popularity
    for j in rated:
        scores[j] = -1e9
    order = np.argsort(-scores)
    return [(int(j), "taste_adj") for j in order[:k]]


def validity_battery(D, split, u, kmax=8):
    """Hole-2: held-out-RATED items + popularity+genre-MATCHED never-rated items.
    SEPARATE from the interview bank (these ARE held-out targets)."""
    kn, ho = split[u]
    rat = dict(D["rat_by_u"][u])
    rated_all = set(rat)
    held_rated = [j for j in ho if rat.get(j, 0) >= 1]          # rated in the held-out half
    rng = np.random.default_rng(1000 + u)
    rng.shuffle(held_rated); held_rated = held_rated[:kmax]
    matched = []
    for j in held_rated:
        tier = D["tier"][j]; pg = D["dgen"][j][0]
        cand = np.where((D["tier"] == tier) & (D["Gmat"][:, GIX.get(pg, GIX["(no genres listed)"])] > 0))[0]
        cand = [int(c) for c in cand if c not in rated_all]
        if cand:
            matched.append((int(rng.choice(cand)), tier, pg))
    held_rated = [(j, D["tier"][j], D["dgen"][j][0]) for j in held_rated]
    return held_rated, matched


# ----------------------------------------------------------------------------- prompts
SYS_ANSWER = (
    "You judge what ONE specific movie viewer could MEANINGFULLY ANSWER in a cold-start preference "
    "interview. You are shown a SAMPLE of this viewer's past ratings (they have seen many more films "
    "than listed). Use general world knowledge about films and about what a person with this history "
    "plausibly knows. Judge ANSWERABILITY, not whether they'd like it.\n"
    "Definition of a meaningful answer PER QUESTION TYPE:\n"
    "  [item]   the viewer has seen the film / holds a real opinion about it.\n"
    "  [concept] the viewer is familiar enough with this kind/aspect of film to state a preference.\n"
    "Do NOT assume famous = answerable by rule; reason about THIS viewer.\n"
    'Output STRICT JSON: {"answers":[{"q":<int index>,"can_answer":"yes"|"no"|"maybe",'
    '"conf":<0..1>}]} . One object per question, no prose.')

SYS_MASK = (
    "You predict what rating ONE specific movie viewer WOULD have given to each listed film, based on a "
    "SAMPLE of their other ratings (they have seen many more films than listed). Ratings are on a 0.5 to "
    "5.0 scale (half-star steps). Predict the viewer's rating for each target film.\n"
    'Output STRICT JSON: {"answers":[{"q":<int index>,"pred_value":<0.5..5.0>,"conf":<0..1>}]} . '
    "One object per film, no prose.")


def profile_text(D, known_ids, rat, exclude=()):
    ex = set(exclude)
    items = [(j, rat[j]) for j in known_ids if j not in ex]
    lines = [f"{D['title'][j]} = {rat[j]:.1f}/5" for j, _ in items]
    return lines


# ----------------------------------------------------------------------------- per-user answerability pass
def run_user_answerability(D, split, u, concepts, item_bank, taste_items, valid_rated, valid_never):
    kn, ho = split[u]; rat = dict(D["rat_by_u"][u])
    known_ids = sorted(kn)
    prof_lines = profile_text(D, known_ids, rat)
    n_shown, N_total = len(prof_lines), len(dict(D["rat_by_u"][u]))

    # assemble the question list; tag each with kind so metrics can separate interview vs validity.
    Q = []   # (kind, meta)
    for ci, name, breadth in concepts:
        Q.append(("concept", dict(name=name, breadth=breadth, ctag=ci)))
    for j, tier in item_bank:
        Q.append(("item", dict(j=j, tier=tier, sub="designed")))
    for j, tier in taste_items:
        Q.append(("item", dict(j=j, tier=D["tier"][j], sub="taste_adj")))
    for j, tier, pg in valid_rated:
        Q.append(("valid_rated", dict(j=j, tier=tier, pg=pg)))
    for j, tier, pg in valid_never:
        Q.append(("valid_never", dict(j=j, tier=tier, pg=pg)))

    # render question text
    lines = []
    for i, (kind, meta) in enumerate(Q):
        if kind == "concept":
            lines.append(f"{i}: [concept] Could this viewer state a preference about \"{meta['name']}\" films?")
        else:
            lines.append(f"{i}: [item] Has this viewer seen / could they give a real opinion on "
                         f"'{D['title'][meta['j']]}'?")
    prof = "; ".join(prof_lines) if prof_lines else "(no ratings shown)"
    user_msg = (f"Viewer's rating sample (size {n_shown} of {N_total} total; they have seen many more "
                f"films than listed):\n{prof}\n\nQuestions:\n" + "\n".join(lines))
    arr = _call_json(SYS_ANSWER, user_msg)
    ans = {}
    for o in arr:
        if isinstance(o, dict) and "q" in o:
            try:
                ans[int(o["q"])] = o
            except (ValueError, TypeError):
                pass
    return Q, ans, (n_shown, N_total)


def run_user_mask(D, split, u, kmax=8):
    """Hole-1 SEPARATE calibration pass: mask known-rated items from the shown profile, predict rating."""
    kn, ho = split[u]; rat = dict(D["rat_by_u"][u])
    known_ids = sorted(kn)
    rng = np.random.default_rng(2000 + u)
    cand = known_ids[:]; rng.shuffle(cand); masked = cand[:kmax]
    prof_lines = profile_text(D, known_ids, rat, exclude=masked)
    n_shown, N_total = len(prof_lines), len(dict(D["rat_by_u"][u]))
    lines = [f"{i}: '{D['title'][j]}'" for i, j in enumerate(masked)]
    prof = "; ".join(prof_lines) if prof_lines else "(no ratings shown)"
    user_msg = (f"Viewer's rating sample (size {n_shown} of {N_total} total; they have seen many more "
                f"films than listed):\n{prof}\n\nPredict this viewer's rating for each target film:\n"
                + "\n".join(lines))
    arr = _call_json(SYS_MASK, user_msg)
    preds = {}
    for o in arr:
        if isinstance(o, dict) and "q" in o and "pred_value" in o:
            try:
                preds[int(o["q"])] = float(o["pred_value"])
            except (ValueError, TypeError):
                pass
    rows = []
    for i, j in enumerate(masked):
        if i in preds:
            rows.append(dict(j=j, true=rat[j], pred=preds[i]))
    return rows


# ----------------------------------------------------------------------------- metrics
def is_yes(o):
    return isinstance(o, dict) and o.get("can_answer") == "yes"   # maybe->refuse (primary)


def compute_metrics(D, per_user):
    # ---- heterogeneity: per-user interview answer-rate spread ----
    rates = []
    for u, rec in per_user.items():
        Q, ans = rec["Q"], rec["ans"]
        iv = [i for i, (k, m) in enumerate(Q) if k in ("concept", "item")]
        yes = sum(1 for i in iv if is_yes(ans.get(i, {})))
        rates.append(yes / max(len(iv), 1))
    rates = np.array(rates)
    hetero = dict(mean=float(rates.mean()), sd=float(rates.std()), min=float(rates.min()),
                  max=float(rates.max()), spread=float(rates.max() - rates.min()),
                  note="mixed-effects logistic can_answer~logpop+breadth+(1|user)+(1|question) is a "
                       "TODO stub for the full gate; mini-pilot reports the raw per-user answer-rate spread")

    # ---- taste-tracking: point-biserial corr between taste-match and can_answer (concepts) ----
    tm, yn = [], []
    for u, rec in per_user.items():
        Q, ans = rec["Q"], rec["ans"]
        dgv = rec["dg_vec"]
        for i, (k, m) in enumerate(Q):
            if k == "concept" and i in ans:
                cv = concept_genre_vec(D, m["ctag"])
                nv = np.linalg.norm(dgv)
                cm = float(dgv @ cv / nv) if nv > 0 else 0.0
                tm.append(cm); yn.append(1 if is_yes(ans[i]) else 0)
    tm, yn = np.array(tm), np.array(yn)
    if len(tm) > 3 and yn.std() > 0 and tm.std() > 0:
        r = float(np.corrcoef(tm, yn)[0, 1])
    else:
        r = float("nan")
    taste = dict(point_biserial_r=r, n=int(len(tm)),
                 mean_tastematch_when_yes=float(tm[yn == 1].mean()) if (yn == 1).any() else None,
                 mean_tastematch_when_no=float(tm[yn == 0].mean()) if (yn == 0).any() else None,
                 note="full gate: mixed-effects taste-match coeff (OR>=1.5/sd or dAUC>=0.05). "
                      "mini-pilot reports raw correlation as plumbing.")

    # ---- VALIDITY GAP (Hole 2): held-out-rated vs matched never-rated, WITHIN pop tiers ----
    tiers = ["famous", "moderate", "obscure"]
    agg = {t: dict(rated_yes=0, rated_n=0, never_yes=0, never_n=0) for t in tiers}
    for u, rec in per_user.items():
        Q, ans = rec["Q"], rec["ans"]
        for i, (k, m) in enumerate(Q):
            if k not in ("valid_rated", "valid_never") or i not in ans:
                continue
            t = m["tier"]
            if t not in agg:
                continue
            y = 1 if is_yes(ans[i]) else 0
            if k == "valid_rated":
                agg[t]["rated_yes"] += y; agg[t]["rated_n"] += 1
            else:
                agg[t]["never_yes"] += y; agg[t]["never_n"] += 1
    validity = {}
    for t in tiers:
        a = agg[t]
        rr = a["rated_yes"] / a["rated_n"] if a["rated_n"] else None
        nr = a["never_yes"] / a["never_n"] if a["never_n"] else None
        gap = (rr - nr) if (rr is not None and nr is not None) else None
        validity[t] = dict(rated_rate=rr, never_rate=nr, gap_pts=(gap * 100 if gap is not None else None),
                           rated_n=a["rated_n"], never_n=a["never_n"])
    validity["_prereg"] = "require >=15pt gap in low/mid tiers (obscure/moderate). one-sided signal."

    # ---- masked-item MAE (Hole 1) ----
    errs, tp = [], []
    for u, rec in per_user.items():
        for row in rec.get("mask", []):
            errs.append(abs(row["pred"] - row["true"])); tp.append((row["true"], row["pred"]))
    if errs:
        tp = np.array(tp)
        mae = float(np.mean(errs))
        corr = float(np.corrcoef(tp[:, 0], tp[:, 1])[0, 1]) if len(tp) > 3 and tp[:, 1].std() > 0 else float("nan")
    else:
        mae, corr = None, None
    masked = dict(llm_mae=mae, llm_pred_true_corr=corr, n=len(errs),
                  note="feeds counterfactual-value fidelity sigma (Hole 4); CF cross-check EASE = full-gate TODO")

    # ---- G2 exploitability: stub (real run on cached grid) ----
    g2 = dict(status="STUB", note="answerability-aware vs blind greedy dNDCG on cached grid; requires the "
              "instrument fold-in loop — computed in the 300-user gate, not the mini-pilot.")
    return dict(heterogeneity=hetero, taste_tracking=taste, validity_gap=validity,
                masked_item_mae=masked, g2_exploitability=g2)


# ----------------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minipilot", action="store_true")
    ap.add_argument("--n_users", type=int, default=15)
    ap.add_argument("--n_concepts", type=int, default=39)
    ap.add_argument("--n_items", type=int, default=30)
    args = ap.parse_args()
    if not args.minipilot:
        print("Refusing to run the full gate. Pass --minipilot (owner review gates the 300-user run).")
        return

    t0 = time.time()
    print(f"[gate] model={MODEL} temp={TEMPERATURE} split_seed={ANSWERER_SPLIT_SEED} "
          f"user_seed={USER_SAMPLE_SEED}", flush=True)
    D = load_data()
    print(f"[data] ni={D['ni']} nu={D['nu']} eval-cohort users={len(D['rat_by_u'])}", flush=True)
    split = build_split(D)
    users = pick_users(D, split, args.n_users)
    print(f"[users] picked {len(users)} (profile-size-tercile x dominant-genre):", flush=True)
    concepts = concept_bank(D, args.n_concepts)
    item_bank = designed_item_bank(D, args.n_items)

    per_user = {}
    for u, cell in users:
        kn, ho = split[u]; rat = dict(D["rat_by_u"][u])
        dg_name, dg_vec = dominant_genre(D, sorted(kn), rat)
        taste_items = taste_adjacent_items(D, sorted(kn), rat, dg_vec, k=6)
        vr, vn = validity_battery(D, split, u, kmax=8)
        print(f"  user {u} cell(size={cell[0]},{cell[1]}) N={len(rat)} known={len(kn)} "
              f"dom={dg_name} valid(rated={len(vr)},never={len(vn)})", flush=True)
        Q, ans, prof = run_user_answerability(D, split, u, concepts, item_bank, taste_items, vr, vn)
        mask = run_user_mask(D, split, u, kmax=8)
        iv = [i for i, (k, m) in enumerate(Q) if k in ("concept", "item")]
        yes = sum(1 for i in iv if is_yes(ans.get(i, {})))
        print(f"    -> interview yes {yes}/{len(iv)} | answers parsed {len(ans)}/{len(Q)} | "
              f"mask preds {len(mask)}", flush=True)
        per_user[u] = dict(cell=cell, Q=Q, ans=ans, mask=mask, dg_vec=dg_vec.tolist(),
                           dom=dg_name, prof=prof)

    # metrics need dg_vec as array
    for u in per_user:
        per_user[u]["dg_vec"] = np.array(per_user[u]["dg_vec"])
    M = compute_metrics(D, per_user)

    print("\n================= MINI-PILOT METRICS (PLUMBING SANITY) =================", flush=True)
    print("HETEROGENEITY (interview answer-rate):", json.dumps(M["heterogeneity"], indent=1), flush=True)
    print("TASTE-TRACKING:", json.dumps(M["taste_tracking"], indent=1), flush=True)
    print("VALIDITY GAP (held-out-rated - matched-never, within pop tier):",
          json.dumps(M["validity_gap"], indent=1), flush=True)
    print("MASKED-ITEM MAE:", json.dumps(M["masked_item_mae"], indent=1), flush=True)
    print("G2:", json.dumps(M["g2_exploitability"], indent=1), flush=True)

    total_min = (time.time() - t0) / 60
    cost = dict(calls=_USAGE["calls"], prompt_tokens=_USAGE["prompt_tokens"],
                completion_tokens=_USAGE["completion_tokens"], usd_estimate=round(_USAGE["usd"], 4),
                usd_per_call=round(_USAGE["usd"] / max(_USAGE["calls"], 1), 5),
                snapshot=_USAGE["snapshot"], price_in_per_m=PRICE_IN_PER_M, price_out_per_m=PRICE_OUT_PER_M,
                wall_min=round(total_min, 2))
    print("\nCOST:", json.dumps(cost, indent=1), flush=True)

    out = dict(config=dict(model=MODEL, snapshot=_USAGE["snapshot"], temperature=TEMPERATURE,
                           answerer_split_seed=ANSWERER_SPLIT_SEED, user_sample_seed=USER_SAMPLE_SEED,
                           n_users=len(users), n_concepts=len(concepts), n_items=len(item_bank),
                           maybe_policy="refuse", dataset="ML-25M", meta=META),
               cost=cost, metrics=M,
               per_user={str(u): dict(cell=list(r["cell"][1:]) if False else r["cell"],
                                      dom=r["dom"], N=len(dict(D["rat_by_u"][u])),
                                      interview_yes=sum(1 for i, (k, m) in enumerate(r["Q"])
                                                        if k in ("concept", "item") and is_yes(r["ans"].get(i, {}))),
                                      interview_n=sum(1 for k, m in r["Q"] if k in ("concept", "item")),
                                      mask_n=len(r["mask"]))
                         for u, r in per_user.items()})
    os.makedirs("experiments", exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"), indent=1, default=str)
    print(f"\n[saved] {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
