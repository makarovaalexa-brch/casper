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
import os, re, json, time, argparse, collections, sys
import numpy as np
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

# gate-only heavy deps (instrument fold-in + stats). Imported lazily-safe at module load.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "instrument2"))
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

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

# ---- full-gate pins (G2 fold-in reuses the certified ML-25M instrument + PHASE4A conventions) ----
RECVAE_CKPT = ".cache/instrument2/ml25m_recvae_d512_best.pt"
ETA = 16.0                                   # belief operator z' = z + eta*a*q (ml25m_gates G8 value)
T_TURNS = 8                                  # interview length (PHASE4A T=8)
N_TASTE_CLUSTERS = 5                         # dominant-taste-cluster stratum (size-tercile x cluster => 15 cells)
GATE_OUT_JSON = "experiments/answerability_gate_ml25m_results.json"
GRID_CACHE = ".cache/instrument2/answerability_grid_ml25m.json"     # resumable per-user judgment grid
GATE_RESULT_MD = "experiments/ANSWERABILITY_GATE_RESULT.md"

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


def pick_users(D, split, n_users, n_clusters=N_TASTE_CLUSTERS):
    """Stratify eval-cohort (te) users by profile-size TERCILE x DOMINANT-TASTE-CLUSTER.

    Per the review (Hole 7 + C3): 1-user-per-genre confounds user identity with taste and kills
    taste-tracking power. Instead we cluster users on their known-profile genre distribution
    (KMeans, k=5) and cross with a profile-size tercile => 15 cells; round-robin fill so each
    populated cell reaches >=~20 users at n_users=300. Cluster label is the taste stratum."""
    from sklearn.cluster import KMeans
    te = [u for u in D["te"].tolist() if u in split]
    # genre-distribution feature per user (over known-half likes)
    feats = np.zeros((len(te), len(GENRES)), np.float64)
    sizes = {}
    for r, u in enumerate(te):
        kn, _ = split[u]; rat = dict(D["rat_by_u"][u])
        _, gvec = dominant_genre(D, sorted(kn), rat)
        feats[r] = gvec
        sizes[u] = len(rat)
    km = KMeans(n_clusters=n_clusters, random_state=USER_SAMPLE_SEED, n_init=10).fit(feats)
    clu = {u: int(km.labels_[r]) for r, u in enumerate(te)}
    qs = np.quantile(list(sizes.values()), [1 / 3, 2 / 3])
    rng = np.random.default_rng(USER_SAMPLE_SEED)
    cells = collections.defaultdict(list)
    for u in te:
        sz = 0 if sizes[u] <= qs[0] else (1 if sizes[u] <= qs[1] else 2)
        cells[(sz, clu[u])].append(u)
    keys = sorted(cells.keys()); rng.shuffle(keys)
    picked, ci = [], 0
    while len(picked) < n_users and keys:
        k = keys[ci % len(keys)]
        if cells[k]:
            u = cells[k].pop(rng.integers(len(cells[k])))
            picked.append((u, k))
        ci += 1
        if ci > 100000:
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


# ============================================================================= FULL GATE
# ---- per-question features (for heterogeneity + taste-tracking) ------------------------
def question_features(D, kind, meta, dg_vec):
    """Return (logpop, breadth_ord, taste_match) for an interview question.
    logpop: item=log(train-like cnt+1); concept=log(tag_mass+1).
    breadth_ord: item famous/moderate/obscure = 2/1/0; concept broad/mid/niche = 2/1/0.
    taste_match: cosine(user known-profile genre dist, question genre vector)."""
    nv = np.linalg.norm(dg_vec)
    if kind == "concept":
        cv = concept_genre_vec(D, meta["ctag"])
        tm = float(dg_vec @ cv / nv) if nv > 0 else 0.0
        logpop = float(np.log(D["concepts"]["mass"][meta["ctag"]] + 1.0))
        breadth = {"broad": 2, "mid": 1, "niche": 0}.get(meta.get("breadth"), 1)
    else:  # item
        j = meta["j"]
        gv = D["Gmat"][j].astype(np.float64); n2 = np.linalg.norm(gv)
        tm = float(dg_vec @ gv / (nv * n2)) if (nv > 0 and n2 > 0) else 0.0
        logpop = float(np.log(D["cnt"][j] + 1.0))
        breadth = {"famous": 2, "moderate": 1, "obscure": 0}.get(D["tier"][j], 1)
    return logpop, breadth, tm


def build_interview_rows(D, per_user):
    """Flatten interview (concept+item) judgments into a design matrix for the mixed model."""
    rows = []
    for u, rec in per_user.items():
        Q, ans, dgv = rec["Q"], rec["ans"], np.asarray(rec["dg_vec"])
        for i, (k, m) in enumerate(Q):
            if k not in ("concept", "item") or i not in ans:
                continue
            logpop, breadth, tm = question_features(D, k, m, dgv)
            qid = f"C:{m['ctag']}" if k == "concept" else f"I:{m['j']}"
            rows.append(dict(user=int(u), qid=qid, y=1 if is_yes(ans[i]) else 0,
                             logpop=logpop, breadth=float(breadth), tm=tm, kind=k))
    return rows


# ---- TEST 1: HETEROGENEITY (crossed random-intercept logistic, EB two-step + permutation) ----
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _solve_intercepts(groups, eta0, y, cap=6.0, iters=40):
    """Per-group logit random intercept b_g solving sum_i[y - sigmoid(eta0+b_g)]=0 (Newton).
    Returns dict g->b and dict g->info(sum p(1-p)) for sampling-variance correction."""
    idx = collections.defaultdict(list)
    for i, g in enumerate(groups):
        idx[g].append(i)
    b, info = {}, {}
    for g, ii in idx.items():
        e = eta0[ii]; yy = y[ii]; bg = 0.0
        for _ in range(iters):
            p = _sigmoid(e + bg)
            grad = np.sum(yy - p); hess = np.sum(p * (1 - p)) + 1e-9
            step = grad / hess; bg += step
            if abs(step) < 1e-6:
                break
        bg = float(np.clip(bg, -cap, cap))
        p = _sigmoid(e + bg)
        b[g] = bg; info[g] = float(np.sum(p * (1 - p)))
    return b, info


def _var_corrected(bvals, infos):
    """Latent-scale variance of estimated intercepts, minus mean sampling variance (1/info)."""
    b = np.array(list(bvals.values()))
    samp = np.array([1.0 / max(infos[g], 1e-6) for g in bvals])
    raw = float(np.var(b, ddof=1)) if len(b) > 1 else 0.0
    corr = raw - float(np.mean(samp))
    return raw, max(corr, 0.0)


def heterogeneity_full(D, per_user, rows, n_perm=300):
    from sklearn.linear_model import LogisticRegression
    y = np.array([r["y"] for r in rows], float)
    Xf = np.column_stack([[r["logpop"] for r in rows], [r["breadth"] for r in rows]]).astype(float)
    Xf = (Xf - Xf.mean(0)) / (Xf.std(0) + 1e-9)
    users = np.array([r["user"] for r in rows]); qids = np.array([r["qid"] for r in rows])
    # fixed-effect logistic (unpenalized) -> eta0 (fixed linear predictor)
    if y.min() == y.max():
        eta0 = np.full(len(y), np.log((y.mean() + 1e-6) / (1 - y.mean() + 1e-6)))
    else:
        lr = LogisticRegression(penalty=None, max_iter=2000).fit(Xf, y)
        eta0 = Xf @ lr.coef_[0] + lr.intercept_[0]
    # EB two-step: user intercepts, then question intercepts on user-offset predictor
    bu, iu = _solve_intercepts(users, eta0, y)
    eta1 = eta0 + np.array([bu[u] for u in users])
    bq, iq = _solve_intercepts(qids, eta1, y)
    _, vu = _var_corrected(bu, iu)
    _, vq = _var_corrected(bq, iq)
    PI2_3 = np.pi ** 2 / 3.0
    icc_user = vu / (vu + vq + PI2_3)
    # permutation test on user labels for the user variance component
    rng = np.random.default_rng(0); ge = 0
    for _ in range(n_perm):
        pu = rng.permutation(users)
        bpu, ipu = _solve_intercepts(pu, eta0, y)
        _, vpu = _var_corrected(bpu, ipu)
        if vpu >= vu:
            ge += 1
    p_perm = (ge + 1) / (n_perm + 1)
    # raw per-user answer-rate spread (interpretability)
    rates = []
    for u, rec in per_user.items():
        Q, ans = rec["Q"], rec["ans"]
        iv = [i for i, (k, m) in enumerate(Q) if k in ("concept", "item")]
        yes = sum(1 for i in iv if is_yes(ans.get(i, {})))
        rates.append(yes / max(len(iv), 1))
    rates = np.array(rates)
    return dict(method="EB two-step crossed random-intercept logistic on (logpop+breadth)-adjusted "
                "answers; latent-scale variance components (statsmodels GLMM unavailable in env); "
                "ICC=var_user/(var_user+var_question+pi^2/3); permutation LRT-surrogate on user labels.",
                var_user_latent=float(vu), var_question_latent=float(vq),
                icc_user=float(icc_user), perm_p_user_var=float(p_perm), n_obs=int(len(y)),
                n_users=int(len(set(users))), n_questions=int(len(set(qids))),
                raw_user_rate_mean=float(rates.mean()), raw_user_rate_sd=float(rates.std()),
                raw_user_rate_min=float(rates.min()), raw_user_rate_max=float(rates.max()),
                cutoff="ICC>=0.05", passed=bool(icc_user >= 0.05 and p_perm < 0.05))


# ---- TEST 2: TASTE-TRACKING (logistic OR/sd + grouped-CV dAUC vs popularity-only) ----
def taste_tracking_full(rows, n_folds=5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    y = np.array([r["y"] for r in rows], float)
    logpop = np.array([r["logpop"] for r in rows]); breadth = np.array([r["breadth"] for r in rows])
    tm = np.array([r["tm"] for r in rows]); users = np.array([r["user"] for r in rows])
    def z(a): return (a - a.mean()) / (a.std() + 1e-9)
    Xfull = np.column_stack([z(logpop), z(breadth), z(tm)])
    Xpop = np.column_stack([z(logpop)])
    # OR per sd of taste-match from the full-data fit
    lr = LogisticRegression(penalty=None, max_iter=2000).fit(Xfull, y)
    or_tm = float(np.exp(lr.coef_[0][2]))
    # grouped 5-fold CV by user -> dAUC(full - pop-only) out of sample
    uniq = np.array(sorted(set(users.tolist()))); rng = np.random.default_rng(0); rng.shuffle(uniq)
    folds = np.array_split(uniq, n_folds); umap = {u: k for k, f in enumerate(folds) for u in f}
    fold_of = np.array([umap[u] for u in users])
    pf, pp, yt = [], [], []
    for k in range(n_folds):
        tr, teM = fold_of != k, fold_of == k
        if y[tr].min() == y[tr].max() or teM.sum() == 0:
            continue
        m1 = LogisticRegression(penalty=None, max_iter=2000).fit(Xfull[tr], y[tr])
        m0 = LogisticRegression(penalty=None, max_iter=2000).fit(Xpop[tr], y[tr])
        pf.append(m1.predict_proba(Xfull[teM])[:, 1]); pp.append(m0.predict_proba(Xpop[teM])[:, 1])
        yt.append(y[teM])
    yt = np.concatenate(yt); pf = np.concatenate(pf); pp = np.concatenate(pp)
    auc_full = float(roc_auc_score(yt, pf)); auc_pop = float(roc_auc_score(yt, pp))
    d_auc = auc_full - auc_pop
    passed = (or_tm >= 1.5) or (d_auc >= 0.05)
    return dict(method="unpenalized logistic; OR per sd from full-data fit; dAUC = grouped-5fold-CV "
                "(split by user) AUC(logpop+breadth+tastematch) - AUC(logpop-only).",
                or_tastematch_per_sd=or_tm, auc_full=auc_full, auc_pop_only=auc_pop, d_auc=float(d_auc),
                n_obs=int(len(y)), cutoff="OR>=1.5/sd OR dAUC>=0.05", passed=bool(passed))


# ---- TEST 4: G2 EXPLOITABILITY (answerability-aware vs -blind greedy on the I2 fold-in) ----
def _ndcg_top10(score, tlike, profset, headmask, tail):
    """Exact NDCG@10 (argpartition top-10 then sort) == ml25m_arena.ndcg_at10, faster."""
    s = score.copy(); s[list(profset)] = -1e9
    if tail:
        s[headmask] = -1e9
        rel = set(t for t in tlike if not headmask[t])
    else:
        rel = set(tlike)
    if not rel:
        return None
    top = np.argpartition(-s, 10)[:10]
    top = top[np.argsort(-s[top])]
    W = 1.0 / np.log2(np.arange(2, 12))
    dcg = sum(W[p] for p, t in enumerate(top) if int(t) in rel)
    idcg = W[:min(10, len(rel))].sum() + 1e-12
    return dcg / idcg


def g2_exploitability(D, split, per_user, n_boot=2000):
    """Upper-bound the adaptive prize from answerability discovery on the certified ML-25M RecVAE.

    For each user: z* = enc(known-half likes) [true taste]; targets = held-out likes; belief operator
    z' = z + ETA*a*q with graded a = cos(z*, q); NDCG@10(full) via the arena metric.
    Candidate questions = interview concepts + designed/taste items (their I2 directions), each tagged
    with the LLM answerability (maybe->refuse).
      Selector A (answerability-AWARE, per-user oracle): greedy T turns, each turn picks the ANSWERABLE
        unused question maximizing NDCG after its update. Upper bound (free answerability table).
      Selector B (answerability-BLIND): ONE fixed T-question schedule chosen by greedy forward selection
        to maximize MEAN NDCG over the cohort; a scheduled question updates a user's belief only if THAT
        user can answer it (blind schedule, realistic per-user answering). Strongest single static schedule.
    dNDCG = A - B (paired, per user); paired bootstrap CI. PASS if CI excludes 0 AND dNDCG>=0.005."""
    import torch
    from recvae import RecVAE
    import ml25m_arena as A
    blob = torch.load(RECVAE_CKPT, map_location="cpu")
    a = blob["args"]; model = RecVAE(a["hidden"], a["latent"], int(D["ni"]))
    model.load_state_dict(blob["model"]); model.eval()
    d = model.decoder.in_features
    W = model.decoder.weight.detach().numpy().astype(np.float64)      # (ni,d)
    bdec = model.decoder.bias.detach().numpy().astype(np.float64)
    Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)        # unit item-decoder-row dirs
    headmask = A.load_arena(seed=ANSWERER_SPLIT_SEED)["headmask"]

    def enc_bag(items):
        x = np.zeros((1, int(D["ni"])), np.float32)
        if items:
            x[0, list(items)] = 1.0
        with torch.no_grad():
            mu, _ = model.encoder(torch.tensor(x), dropout_rate=0.0)
        return mu.numpy()[0].astype(np.float64)

    # concept directions (member-bag encode, PHASE4A W2 convention) for judged concepts
    item_tag = D["concepts"]["item_tag"]; Mbag = 50
    concept_dir_cache = {}
    def concept_dir(ctag):
        if ctag not in concept_dir_cache:
            rel = item_tag[:, ctag].astype(np.float64)
            top = np.argpartition(-rel, Mbag)[:Mbag]
            b = np.zeros(int(D["ni"]), np.float32); b[top] = rel[top]; b /= b.sum() + 1e-9
            with torch.no_grad():                       # relevance-weighted member-bag encode (W2 conv.)
                mu, _ = model.encoder(torch.tensor(b[None, :]), dropout_rate=0.0)
            z = mu.numpy()[0].astype(np.float64)
            concept_dir_cache[ctag] = z / (np.linalg.norm(z) + 1e-9)
        return concept_dir_cache[ctag]

    # assemble per-user: z*, targets, candidate (dir, answerable) list
    U = []
    for u, rec in per_user.items():
        kn, ho = split[u]; rat = dict(D["rat_by_u"][u])
        prof = [j for j in kn if rat.get(j, 0) >= 4]
        tlike = set(j for j in ho if rat.get(j, 0) >= 4)
        if not prof or not tlike:
            continue
        zstar = enc_bag(prof); nz = np.linalg.norm(zstar) + 1e-9
        cand_dir, cand_ans, cand_key = [], [], []
        for i, (k, m) in enumerate(rec["Q"]):
            if k == "concept":
                q = concept_dir(m["ctag"]); key = f"C:{m['ctag']}"
            elif k == "item":
                q = Wn[m["j"]]; key = f"I:{m['j']}"
            else:
                continue
            cand_dir.append(q); cand_ans.append(is_yes(rec["ans"].get(i, {}))); cand_key.append(key)
        if not cand_dir:
            continue
        U.append(dict(u=int(u), zstar=zstar, nz=nz, prof=set(kn), tlike=tlike,
                      dir=np.array(cand_dir), ans=np.array(cand_ans, bool), key=cand_key))
    if not U:
        return dict(status="FAIL", error="no eligible users for G2 fold-in (empty prof/targets)")

    def decode_scores(Z):    # Z (n,d) -> (n,ni)
        return Z @ W.T + bdec

    # ---- Selector A: per-user answerability-aware greedy oracle ----
    a_ndcg = {}
    for rec in U:
        zstar, nz = rec["zstar"], rec["nz"]; dirs = rec["dir"]; ok = np.where(rec["ans"])[0]
        z = np.zeros(d); used = set()
        for _ in range(T_TURNS):
            avail = [c for c in ok if c not in used]
            if not avail:
                break
            # candidate next-beliefs
            a_vals = (dirs[avail] @ zstar) / nz                 # graded answers cos(z*,q)
            Znext = z[None, :] + ETA * a_vals[:, None] * dirs[avail]
            S = decode_scores(Znext)
            best, bi = -1.0, None
            for r, c in enumerate(avail):
                n = _ndcg_top10(S[r], rec["tlike"], rec["prof"], headmask, False)
                if n is not None and n > best:
                    best, bi = n, c
            if bi is None:
                break
            a_v = float((dirs[bi] @ zstar) / nz); z = z + ETA * a_v * dirs[bi]; used.add(bi)
        S = decode_scores(z[None, :])[0]
        a_ndcg[rec["u"]] = _ndcg_top10(S, rec["tlike"], rec["prof"], headmask, False)

    # ---- Selector B: single blind schedule (greedy forward, cohort-mean NDCG; per-user answer gating) ----
    # global candidate key universe (union of concept + designed-item keys shared across users;
    # taste-adjacent item keys are per-user so excluded from the shared blind schedule pool)
    keycount = collections.Counter(k for rec in U for k in rec["key"])
    shared_keys = [k for k, c in keycount.items() if c >= max(2, int(0.5 * len(U)))]
    # map key -> direction (any user's copy; concept/decoder-row dirs are user-independent)
    key_dir = {}
    for rec in U:
        for c, k in enumerate(rec["key"]):
            if k in shared_keys and k not in key_dir:
                key_dir[k] = rec["dir"][c]
    # per-user: can-answer lookup + current belief state during schedule build
    for rec in U:
        rec["_kset"] = {k: rec["ans"][c] for c, k in enumerate(rec["key"])}
    def cohort_mean_ndcg(schedule):
        tot, m = 0.0, 0
        for rec in U:
            z = np.zeros(d)
            for k in schedule:
                if rec["_kset"].get(k, False):                  # only if THIS user can answer
                    q = key_dir[k]; av = float((q @ rec["zstar"]) / rec["nz"]); z = z + ETA * av * q
            S = decode_scores(z[None, :])[0]
            n = _ndcg_top10(S, rec["tlike"], rec["prof"], headmask, False)
            if n is not None:
                tot += n; m += 1
        return tot / max(m, 1)
    schedule, pool = [], list(shared_keys)
    for _ in range(T_TURNS):
        best, bk = -1.0, None
        for k in pool:
            if k in schedule:
                continue
            v = cohort_mean_ndcg(schedule + [k])
            if v > best:
                best, bk = v, k
        if bk is None:
            break
        schedule.append(bk)
    # final per-user B ndcg under the fixed blind schedule
    b_ndcg = {}
    for rec in U:
        z = np.zeros(d)
        for k in schedule:
            if rec["_kset"].get(k, False):
                q = key_dir[k]; av = float((q @ rec["zstar"]) / rec["nz"]); z = z + ETA * av * q
        S = decode_scores(z[None, :])[0]
        b_ndcg[rec["u"]] = _ndcg_top10(S, rec["tlike"], rec["prof"], headmask, False)

    keys = [u for u in a_ndcg if a_ndcg[u] is not None and b_ndcg.get(u) is not None]
    da = np.array([a_ndcg[u] for u in keys]); db = np.array([b_ndcg[u] for u in keys])
    diff = da - db; n = len(diff)
    rng = np.random.default_rng(0)
    boot = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    dmean = float(diff.mean())
    passed = (lo > 0) and (dmean >= 0.005)
    return dict(method="answerability-AWARE per-user greedy oracle (A) vs single BLIND greedy schedule (B) "
                "on the certified ML-25M RecVAE-d512 fold-in; graded answer a=cos(z*,q), z'=z+16*a*q, T=8, "
                "NDCG@10 full; paired user bootstrap.",
                dndcg_mean=dmean, ci95=[float(lo), float(hi)], p_gt0=float((boot > 0).mean()),
                A_mean_ndcg=float(da.mean()), B_mean_ndcg=float(db.mean()), n_users=int(n),
                blind_schedule=schedule, blind_schedule_len=len(schedule),
                cutoff="CI excludes 0 AND dNDCG>=0.005", passed=bool(passed))


# ============================================================================= GATE DRIVER
def run_gate(n_users, n_concepts, n_items, chunk=10, collect_only=False, analyze_only=False, limit=0):
    t0 = time.time()
    print(f"[GATE] model={MODEL} temp={TEMPERATURE} split_seed={ANSWERER_SPLIT_SEED} "
          f"user_seed={USER_SAMPLE_SEED} n_users={n_users} "
          f"{'COLLECT' if collect_only else 'ANALYZE' if analyze_only else 'FULL'}"
          f"{f' limit={limit}' if limit else ''}", flush=True)
    D = load_data()
    print(f"[data] ni={D['ni']} nu={D['nu']} eval-cohort users={len(D['rat_by_u'])}", flush=True)
    split = build_split(D)
    users = pick_users(D, split, n_users)
    cell_hist = collections.Counter(k for _, k in users)
    print(f"[users] picked {len(users)}; cells (size,cluster)->count: {dict(sorted(cell_hist.items()))}",
          flush=True)
    concepts = concept_bank(D, n_concepts)
    item_bank = designed_item_bank(D, n_items)

    # resumable grid cache
    grid = {}
    if os.path.exists(GRID_CACHE):
        blob = json.load(open(GRID_CACHE))
        if blob.get("split_seed") == ANSWERER_SPLIT_SEED and blob.get("model") == MODEL:
            grid = blob.get("users", {})
            print(f"[cache] resuming from {GRID_CACHE}: {len(grid)} users cached", flush=True)

    per_user = {}
    new_this_call = 0
    for n, (u, cell) in enumerate(users):
        kn, ho = split[u]; rat = dict(D["rat_by_u"][u])
        dg_name, dg_vec = dominant_genre(D, sorted(kn), rat)
        if str(u) in grid:  # replay cached judgments (no LLM call)
            g = grid[str(u)]
            Q = [(k, m) for k, m in g["Q"]]
            ans = {int(k): v for k, v in g["ans"].items()}
            mask = g["mask"]; prof = tuple(g["prof"])
        else:
            if analyze_only:
                continue  # analyze pass only uses cached users
            if limit and new_this_call >= limit:
                continue
            taste_items = taste_adjacent_items(D, sorted(kn), rat, dg_vec, k=6)
            vr, vn = validity_battery(D, split, u, kmax=8)
            Q, ans, prof = run_user_answerability(D, split, u, concepts, item_bank, taste_items, vr, vn)
            mask = run_user_mask(D, split, u, kmax=8)
            grid[str(u)] = dict(Q=[[k, {kk: (int(vv) if isinstance(vv, np.integer) else vv)
                                         for kk, vv in m.items()}] for k, m in Q],
                                ans={str(i): o for i, o in ans.items()}, mask=mask, prof=list(prof),
                                cell=list(cell), dom=dg_name)
            new_this_call += 1
            iv = [i for i, (k, m) in enumerate(Q) if k in ("concept", "item")]
            yes = sum(1 for i in iv if is_yes(ans.get(i, {})))
            print(f"  [{len(grid)} cached | +{new_this_call} this call] user {u} cell{cell} "
                  f"N={len(rat)} dom={dg_name} interview yes {yes}/{len(iv)} parsed {len(ans)} "
                  f"| running ${_USAGE['usd']:.3f}", flush=True)
            if new_this_call % chunk == 0:
                json.dump(dict(split_seed=ANSWERER_SPLIT_SEED, model=MODEL, snapshot=_USAGE["snapshot"],
                               users=grid), open(GRID_CACHE, "w"))
                print(f"    [cache] flushed {len(grid)} users -> {GRID_CACHE}", flush=True)
        per_user[u] = dict(cell=cell, Q=Q, ans=ans, mask=mask, dg_vec=dg_vec, dom=dg_name, prof=prof)

    # always flush cache after a collection pass; accumulate cumulative usage across chunked calls
    USAGE_SIDECAR = GRID_CACHE.replace(".json", "_usage.json")
    if not analyze_only:
        json.dump(dict(split_seed=ANSWERER_SPLIT_SEED, model=MODEL, snapshot=_USAGE["snapshot"],
                       users=grid), open(GRID_CACHE, "w"))
        cum = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "usd": 0.0}
        if os.path.exists(USAGE_SIDECAR):
            cum = json.load(open(USAGE_SIDECAR))
        if new_this_call > 0:  # only add this chunk's fresh usage
            cum["calls"] += _USAGE["calls"]; cum["prompt_tokens"] += _USAGE["prompt_tokens"]
            cum["completion_tokens"] += _USAGE["completion_tokens"]; cum["usd"] += _USAGE["usd"]
            cum["snapshot"] = _USAGE["snapshot"]
            json.dump(cum, open(USAGE_SIDECAR, "w"), indent=1)
            print(f"[usage-cum] total {cum['calls']} calls ${cum['usd']:.4f} -> {USAGE_SIDECAR}", flush=True)
    n_cached = sum(1 for u, _ in users if str(u) in grid)
    print(f"[cache] {n_cached}/{len(users)} users cached ({new_this_call} new this call); "
          f"cost so far ${_USAGE['usd']:.3f} ({_USAGE['calls']} calls)", flush=True)
    if collect_only:
        print(f"[COLLECT] done this chunk. {'ALL CACHED - ready for --analyze_only' if n_cached==len(users) else 'RERUN --collect_only to continue.'}", flush=True)
        return
    if n_cached < len(users):
        print(f"[GATE] not all users cached ({n_cached}/{len(users)}); run --collect_only more. Aborting analysis.", flush=True)
        return

    print("\n[GATE] all judgments collected; computing the four gate tests ...", flush=True)
    rows = build_interview_rows(D, per_user)
    het = heterogeneity_full(D, per_user, rows)
    print("  heterogeneity done:", het["icc_user"], het["perm_p_user_var"], flush=True)
    taste = taste_tracking_full(rows)
    print("  taste-tracking done:", taste["or_tastematch_per_sd"], taste["d_auc"], flush=True)
    valm = compute_metrics(D, per_user)  # reuse validity-gap + masked-MAE machinery
    validity = valm["validity_gap"]; masked = valm["masked_item_mae"]
    print("  validity/masked done", flush=True)
    print("  running G2 fold-in (this is the heavy step) ...", flush=True)
    g2 = g2_exploitability(D, split, per_user)
    print("  G2 done:", g2.get("dndcg_mean"), g2.get("ci95"), flush=True)

    # validity-gap pass: >=15pt in low/mid tiers (obscure/moderate), among tiers with data
    vg_low_mid = {t: validity[t]["gap_pts"] for t in ("moderate", "obscure")
                  if validity[t]["gap_pts"] is not None and validity[t]["rated_n"] >= 5}
    validity_pass = any(g >= 15.0 for g in vg_low_mid.values()) if vg_low_mid else False

    verdict = dict(heterogeneity=het["passed"], taste_tracking=taste["passed"],
                   validity_gap=validity_pass, g2_exploitability=g2.get("passed", False))
    overall = all(verdict.values())

    total_min = (time.time() - t0) / 60
    # measured LLM cost = cumulative collection usage (analyze pass itself makes 0 calls; all cached)
    USAGE_SIDECAR = GRID_CACHE.replace(".json", "_usage.json")
    if os.path.exists(USAGE_SIDECAR):
        cum = json.load(open(USAGE_SIDECAR))
        calls, ptok, ctok, usd = cum["calls"], cum.get("prompt_tokens"), cum.get("completion_tokens"), cum["usd"]
        snap = cum.get("snapshot", _USAGE["snapshot"]); cost_note = cum.get("note", "")
    else:
        calls, ptok, ctok, usd = _USAGE["calls"], _USAGE["prompt_tokens"], _USAGE["completion_tokens"], _USAGE["usd"]
        snap = _USAGE["snapshot"]; cost_note = "usd = documented-rate estimate; tokens exact"
    cost = dict(calls=calls, prompt_tokens=ptok, completion_tokens=ctok, usd_estimate=round(usd, 4),
                usd_per_call=round(usd / max(calls, 1), 5), snapshot=snap,
                price_in_per_m=PRICE_IN_PER_M, price_out_per_m=PRICE_OUT_PER_M,
                analyze_wall_min=round(total_min, 2), note=cost_note)
    resolved_snapshot = _USAGE["snapshot"] or snap  # analyze pass makes 0 calls -> use collection snapshot

    out = dict(config=dict(model=MODEL, snapshot=resolved_snapshot, temperature=TEMPERATURE,
                           answerer_split_seed=ANSWERER_SPLIT_SEED, user_sample_seed=USER_SAMPLE_SEED,
                           n_users=len(users), n_concepts=len(concepts), n_items=len(item_bank),
                           maybe_policy="refuse", dataset="ML-25M", meta=META, recvae=RECVAE_CKPT,
                           eta=ETA, T=T_TURNS, n_taste_clusters=N_TASTE_CLUSTERS,
                           cell_histogram={str(k): v for k, v in cell_hist.items()}),
               cost=cost,
               tests=dict(heterogeneity=het, taste_tracking=taste, validity_gap=validity,
                          validity_gap_pass=validity_pass, masked_item_mae=masked, g2_exploitability=g2),
               verdict=dict(per_test=verdict, overall_pass=bool(overall)))
    os.makedirs("experiments", exist_ok=True)
    json.dump(out, open(GATE_OUT_JSON, "w"), indent=1, default=str)
    print(f"\n[saved] {GATE_OUT_JSON}", flush=True)
    write_gate_md(out)
    print(f"[saved] {GATE_RESULT_MD}", flush=True)

    print("\n================= GATE VERDICT =================", flush=True)
    print(f"heterogeneity  ICC={het['icc_user']:.3f} (p={het['perm_p_user_var']:.3f})  "
          f"-> {'PASS' if verdict['heterogeneity'] else 'FAIL'}", flush=True)
    print(f"taste-tracking OR/sd={taste['or_tastematch_per_sd']:.2f} dAUC={taste['d_auc']:.3f}  "
          f"-> {'PASS' if verdict['taste_tracking'] else 'FAIL'}", flush=True)
    print(f"validity-gap   low/mid={vg_low_mid}  -> {'PASS' if verdict['validity_gap'] else 'FAIL'}",
          flush=True)
    print(f"G2 dNDCG={g2.get('dndcg_mean')} CI={g2.get('ci95')}  "
          f"-> {'PASS' if verdict['g2_exploitability'] else 'FAIL'}", flush=True)
    print(f"\nOVERALL ADAPTIVE-CLAIM GATE: {'PASS' if overall else 'FAIL'}", flush=True)
    print(f"COST: {cost['calls']} calls ${cost['usd_estimate']} "
          f"(analyze wall {cost['analyze_wall_min']} min)", flush=True)
    return out


def write_gate_md(out):
    t = out["tests"]; v = out["verdict"]; c = out["config"]; cost = out["cost"]
    het, taste, g2 = t["heterogeneity"], t["taste_tracking"], t["g2_exploitability"]
    vg = t["validity_gap"]
    def yn(b): return "**PASS**" if b else "FAIL"
    L = []
    L.append("# ANSWERABILITY GATE - RESULT (ML-25M, 300-user run)\n")
    L.append(f"Date 2026-07-06. Pre-registration: `answerability_PLAN.md` sec.B (FROZEN cutoffs) + "
             f"`answerability_design_review_2026-07-06.md` (6 holes). Script "
             f"`scripts/llm_answerability_gate.py --gate`. This is the owner-approved decision spend.\n")
    L.append("## Config (resolved snapshot + pins)\n")
    L.append(f"- Model **{c['model']}** -> resolved snapshot **{c['snapshot']}**, temperature "
             f"**{c['temperature']}** (pinned). `maybe`->refuse (primary).\n")
    L.append(f"- Dataset ML-25M; meta `{c['meta']}`; instrument `{c['recvae']}`; belief eta={c['eta']}, "
             f"T={c['T']}.\n")
    L.append(f"- Answerer split seed **{c['answerer_split_seed']}** (fixed, cached); user sample seed "
             f"{c['user_sample_seed']}; **{c['n_users']} users** stratified size-tercile x "
             f"{c['n_taste_clusters']} taste-clusters.\n")
    L.append(f"- Question bank/user: {c['n_concepts']} breadth-tiered concepts + {c['n_items']} designed "
             f"items + 6 taste-adjacent + validity (held-out-rated / matched never-rated) + masked-rated.\n")
    L.append(f"\n## Measured cost\n{cost['calls']} LLM calls (2/user over 300 users), "
             f"**${cost['usd_estimate']}** total (documented-rate estimate; per-call response.usage exact), "
             f"${cost['usd_per_call']}/call. Analyze pass (tests+G2) makes 0 LLM calls "
             f"(all-cached grid), wall {cost['analyze_wall_min']} min. {cost.get('note','')}\n")
    L.append("\n## The four pre-registered gate tests\n")
    L.append("| Test | Cutoff | Measured | Verdict |")
    L.append("|---|---|---|---|")
    L.append(f"| Heterogeneity | ICC>=0.05 (+perm p<.05) | ICC={het['icc_user']:.3f}, "
             f"perm p={het['perm_p_user_var']:.3f} (var_user={het['var_user_latent']:.3f}, "
             f"var_q={het['var_question_latent']:.3f}) | {yn(v['per_test']['heterogeneity'])} |")
    L.append(f"| Taste-tracking | OR>=1.5/sd OR dAUC>=0.05 | OR/sd={taste['or_tastematch_per_sd']:.2f}, "
             f"dAUC={taste['d_auc']:.3f} (full {taste['auc_full']:.3f} vs pop {taste['auc_pop_only']:.3f}) "
             f"| {yn(v['per_test']['taste_tracking'])} |")
    vgs = "; ".join(f"{tt}: {vg[tt]['gap_pts']:.1f}pt (n={vg[tt]['rated_n']})"
                    for tt in ("famous", "moderate", "obscure")
                    if vg[tt]["gap_pts"] is not None)
    L.append(f"| Validity gap (Hole 2) | >=15pt low/mid tier | {vgs} | {yn(v['per_test']['validity_gap'])} |")
    L.append(f"| G2 exploitability (Hole 3) | CI excl 0 AND dNDCG>=0.005 | dNDCG={g2.get('dndcg_mean'):.4f}, "
             f"CI95={[round(x,4) for x in g2.get('ci95',[0,0])]} (A={g2.get('A_mean_ndcg'):.4f} "
             f"B={g2.get('B_mean_ndcg'):.4f}, n={g2.get('n_users')}) | {yn(v['per_test']['g2_exploitability'])} |")
    L.append(f"\n## VERDICT\n\n**ADAPTIVE-CLAIM GATE = heterogeneity AND taste-tracking AND validity-gap "
             f"AND G2 = {'PASS' if v['overall_pass'] else 'FAIL'}.**\n")
    if not v["overall_pass"]:
        failed = [k for k, b in v["per_test"].items() if not b]
        L.append(f"Failed: {', '.join(failed)}. Per pre-registration (PLAN sec.D/E, review Hole 3) a fail "
                 f"means the flagship ships the **STATIC channel map** and the adaptive clause is cut; "
                 f"the gate result is a citable finding either way. STOP-ON-BLOCKER: no weaker test "
                 f"substituted.\n")
    L.append("\n## Method notes\n")
    L.append(f"- Heterogeneity: {het['method']}\n")
    L.append(f"- Taste-tracking: {taste['method']}\n")
    L.append(f"- G2: {g2['method']}\n")
    L.append(f"- Masked-rated-item MAE (Hole 1/4, fidelity sigma): LLM MAE={t['masked_item_mae']['llm_mae']} "
             f"stars, pred/true corr={t['masked_item_mae']['llm_pred_true_corr']}, "
             f"n={t['masked_item_mae']['n']}. Independent-CF cross-check SKIPPED (EASE on 18,430 items = "
             f"a dense 18k^2 Gram inverse, memory-heavy; LLM predictor is the pre-registered primary and "
             f"value is sensitivity-only per Hole-1 rule).\n")
    L.append("- Cross-family spot-check: SKIPPED (no 2nd model-family provider configured in env; only "
             "OPENAI_API_KEY present). Judge-disagreement rate remains an unmeasured uncertainty; the "
             "claim boundary already restricts to 'one LLM judge + a-priori structural rule, human study "
             "pending' (PLAN sec.E).\n")
    L.append("\n## Honest caveats (the de-risk-not-de-circularize boundary)\n")
    L.append("- The LLM judge removes SELF-AUTHORED circularity (no designer-written rule to exploit) but "
             "NOT model-prior dependence: the judge shares text-sources with the world RecVAE models "
             "(Hole 1 shared-prior). Claim boundary for every paper: 'results hold under two external "
             "answer models (LLM judge + a-priori structural rule); the LLM judge is validated against "
             "human answers in a pending human study.' Never 'realistic users'/'human-level'.\n")
    L.append("- G2 dNDCG is an UPPER BOUND on the adaptive prize (Selector A gets the answerability table "
             "for free; a real agent must spend turns learning it). It bounds, not estimates, the agent gain.\n")
    L.append("- Validity gap is a ONE-SIDED signal (never-rated != can't-answer): a positive low/mid-tier "
             "gap proves user-specific signal; a zero gap is ambiguous at high popularity. Only low/mid "
             "tiers are diagnostic.\n")
    L.append("- Single fixed answerer split (seed 123): split-level variance is unmeasured here "
             "(Hole 5 split-sensitivity is a separate later pass, per PLAN sec.D step 4).\n")
    open(GATE_RESULT_MD, "w", encoding="utf-8").write("\n".join(L) + "\n")


# ----------------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minipilot", action="store_true")
    ap.add_argument("--gate", action="store_true", help="owner-approved 300-user ML-25M gate")
    ap.add_argument("--collect_only", action="store_true", help="only collect+cache LLM grid (resumable)")
    ap.add_argument("--analyze_only", action="store_true", help="compute tests+G2 from cached grid")
    ap.add_argument("--limit", type=int, default=0, help="max NEW LLM users this call (chunking)")
    ap.add_argument("--n_users", type=int, default=15)
    ap.add_argument("--n_concepts", type=int, default=39)
    ap.add_argument("--n_items", type=int, default=30)
    args = ap.parse_args()
    if args.gate or args.collect_only or args.analyze_only:
        nu = args.n_users if args.n_users != 15 else 300
        run_gate(nu, args.n_concepts, args.n_items, collect_only=args.collect_only,
                 analyze_only=args.analyze_only, limit=args.limit)
        return
    if not args.minipilot:
        print("Refusing to run the full gate. Pass --minipilot or --gate.")
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
